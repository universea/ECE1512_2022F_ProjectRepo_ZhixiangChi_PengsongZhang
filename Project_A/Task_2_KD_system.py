import timm
import torch.nn as nn
import torch
import os
import torch.utils.data as data
from torchvision import transforms
import csv
import cv2
import argparse
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score
from torchmetrics.functional import auc
import logging
import time
from sklearn.metrics import f1_score, auc, roc_curve
import random
import numpy as np
import copy
import torch.nn.functional as F

class Mhist(data.Dataset):
    def __init__(self, root, data_info, transform=None):
        self.data_info = data_info
        self.transform = transform
        self.root = root
        self.label_map = {'HP': 0, 'SSA': 1}

    def opencv_loader(self, info):

        im = cv2.cvtColor(cv2.imread(os.path.join(self.root, 'images', info[0])), cv2.COLOR_BGR2RGB)
        label = self.label_map[info[1]]
        return im, label

    def __getitem__(self, index):

        im, label = self.opencv_loader(self.data_info[index])

        if self.transform is not None:
            im = self.transform(im)

        return im, label

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.data_info)



class KD():
    def __init__(self, args):
        self.args = args
        self.teacher = timm.create_model(self.args.teacher, num_classes=self.args.num_classes).cuda()
        self.student = timm.create_model(self.args.student, num_classes=self.args.num_classes).cuda()
        self.assistant = timm.create_model(self.args.assistant, num_classes=self.args.num_classes).cuda()
        self.model_files = ['resnetv2_50_a1h-000cdf49.pth', 'mobilenetv2_100_ra-b33bc2c4.pth', 'resnet18-5c106cde.pth']
        self.log_list = []
        self.set_dataloaders()
        self.CE_loss = nn.CrossEntropyLoss()
        self.KL_loss = nn.KLDivLoss()
        self.load_feature_extractors()
        self.student_copy = copy.deepcopy(self.student)
        self.assistant_copy = copy.deepcopy(self.assistant)
        self.f1 = BinaryF1Score().cuda()
        self.scratch_lr = self.args.scratch_LR


    def load_feature_extractors(self):

        models = [self.teacher, self.student, self.assistant]
        for model, name in zip(models, self.model_files):
            ckpt_model = torch.load(os.path.join('./pretrained_models', name))
            model_dict = model.state_dict()
            ckpt_model = {k: v for k, v in ckpt_model.items() if 'fc' not in k and 'classifier' not in k}
            model_dict.update(ckpt_model)
            model.load_state_dict(model_dict)
            print('number of param loaded: ', len(ckpt_model))

    def re_initialize_student(self):
        self.student = copy.deepcopy(self.student_copy)

    def re_initialize_assistant(self):
        self.assistant = copy.deepcopy(self.assistant_copy)


    def log_print(self, message):
        print(message)
        self.log_list.append(message)

    def set_dataloaders(self):
        test_info = []
        train_info = []
        with open(os.path.join(self.args.data_root, 'annotations.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            next(reader)
            for row in reader:
                info = row[0].split(',')
                if info[3] == 'train':
                    train_info.append(info)
                else:
                    test_info.append(info)

        train_trans = transforms.Compose(
            # [transforms.ToTensor(), transforms.RandomHorizontalFlip(),
             [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4866, 0.4409), (0.267, 0.256, 0.276))])
             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
             # ])
        test_trans = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.267, 0.256, 0.276))])
             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
             # ])

        train_set = Mhist(self.args.data_root, train_info, transform=train_trans)
        test_set = Mhist(self.args.data_root, test_info, transform=test_trans)

        self.trainloader = data.DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.testloader = data.DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        train_set_info = train_set.__repr__()
        test_set_info = test_set.__repr__()
        self.log_print(train_set_info)
        self.log_print(test_set_info)

    def set_optimizer(self, model, fintune=True):

        if fintune:
            lr_decay = [10]
            lr = self.args.init_LR
            train_param = []
            for name, param in model.named_parameters():
                if 'fc' in name or 'classifier' in name:
                    train_param += [{'params': param, 'lr': lr}]
        else:
            lr_decay = [10, 20, 30]
            lr = self.scratch_lr
            train_param = list(model.parameters())


        optimizer = torch.optim.Adam(train_param, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay, gamma=self.args.lr_factor)

        return optimizer, scheduler

    def KL_loss_comp(self, teacher, student, temp):

        return self.KL_loss(torch.sigmoid(student / temp), torch.sigmoid(teacher / temp)) * temp ** 2

    def save_list_to_txt(self, name):
        f = open(name, mode='w')
        for item in self.log_list:
            f.write(str(item) + '\n')
        f.close()

    def eval(self, model, dataloader):
        model.eval()
        loss = 0.0
        preds = torch.tensor([]).cuda()
        labels = torch.tensor([]).cuda()
        for idx, data in enumerate(dataloader, 0):
            im, label = data[0].cuda(), data[1].cuda()
            output = model(im)
            loss += self.CE_loss(output, label).item()
            _, pred = output.max(1)
            preds = torch.cat((preds, pred))
            labels = torch.cat((labels, label))
        acc = (preds.eq(labels).sum() / preds.size(0)).item()
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        fpr, tpr, thresholds = roc_curve(labels,preds)
        AUC = auc(fpr, tpr)
        f1 = f1_score(labels, preds)

        return loss/(idx+1), AUC, f1, acc

    def train(self, model, temp, is_distill=False, teachers=[], use_assistant=False, finetune=True, record_name=False):

        record = [[], []]
        if is_distill and len(teachers) == 0:
            teachers.append(self.teacher)
        loss_factor = 1 / (len(teachers) + 1)
        # Create Adam optimizer
        optimizer, sche = self.set_optimizer(model, finetune)
        if finetune:
            model.eval()
        else:
            model.train()

        for epoch in range(self.args.epochs):
            tqdm_gen = tqdm(self.trainloader)

            for trainIndex, trainData in enumerate(tqdm_gen, 0):
                optimizer.zero_grad()
                im, label = trainData[0].cuda(), trainData[1].cuda()
                model_out = model(im)
                # Primary cross-entropy loss
                loss = loss_factor * self.CE_loss(model_out, label)
                # Add KL loss for distillation process
                if is_distill:
                    for teacher in teachers:
                        loss += loss_factor * self.KL_loss_comp(teacher(im), model_out, temp)

                loss.backward()
                optimizer.step()
                tqdm_gen.set_description('Epoch: {}, LR: {:.4f}, train_loss: {:.4f}.'.format(
                    epoch, sche.get_lr()[0], loss.item()))

            sche.step()

            train_loss, train_auc, train_f1, train_acc = self.eval(model, self.trainloader)
            test_loss, test_auc, test_f1, test_acc = self.eval(model, self.testloader)
            message = 'Evaluation at Epoch {}/{}, train_(loss, auc, f1, acc) = {:.3f}, {:.3f}, {:.3f}, {:.2f}' \
                      ', test_(loss, auc, f1, acc) = {:.3f}, {:.3f}, {:.3f}, {:.2f}'.format(epoch, self.args.epochs,
                                                                             train_loss, train_auc, train_f1, train_acc,
                                                                             test_loss, test_auc, test_f1, test_acc)
            record[0].append([train_loss, train_auc, train_f1, train_acc])
            record[1].append([test_loss, test_auc, test_f1, test_acc])

            self.log_print(message)
        if record_name != False:
            np.save(record_name + '.npy', record)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch_LR', default=0.0001, type=float, help='Learning rate for scratch')
    parser.add_argument('--init_LR', default=0.001, type=float, help='Learning rate for finetune')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/data/datasets/mhist_dataset', type=str, help='Root directory for MHIST dataset')
    parser.add_argument('--teacher', default='resnetv2_50', type=str, help='Name for teacher model in TIMM model zoo')
    parser.add_argument('--assistant', default='resnet18', type=str, help='Name for assistant model in TIMM model zoo')
    parser.add_argument('--student', default='mobilenetv2_100', type=str, help='Name for student model in TIMM model zoo')
    parser.add_argument('--num_classes', default='2', type=int, help='Number of classes')
    parser.add_argument('--epochs', default='30', type=int, help='Number of epochs')
    parser.add_argument('--lr_factor', default='0.1', type=float, help='Learning rate decay factor')

    parser.add_argument('--student_scratch', action='store_true')
    parser.add_argument('--student_finetune', action='store_true')
    parser.add_argument('--student_KD', action='store_true')
    parser.add_argument('--use_assistant', action='store_true')
    parser.add_argument('--use_both_teachers', action='store_true')
    parser.add_argument('--student_temp_values', action='store_true')
    parser.add_argument('--assistant_temp_values', action='store_true')


    parser.add_argument('--complete', action='store_true')

    the_args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    temp = 4
    KD_system = KD(the_args)



    if the_args.student_scratch:
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            KD_system.log_print('*' * 10 + ' Training student from scratch, initla LR = {}'.format(lr) + '*' * 10)
            KD_system.scratch_lr = lr
            KD_system.student = timm.create_model('mobilenetv2_100', num_classes=2).cuda()
            KD_system.train(model=KD_system.student, temp=temp, finetune=False, record_name='scratch_lr_{}'.format(lr))

    if the_args.student_finetune or the_args.complete:
        KD_system.log_print('*' * 10 + ' Re-initialize student model' + '*' * 10)
        KD_system.re_initialize_student()
        KD_system.log_print('*' * 10 + ' Fine-tuning student o/w KD' + '*' * 10)
        KD_system.train(model=KD_system.student, temp=temp, finetune=True, record_name='student_FineTune_NO_KD')


    if the_args.student_KD or the_args.complete:
        KD_system.log_print('*' * 10 + ' Fine-tuning teacher model' + '*' * 10)
        KD_system.train(model=KD_system.teacher, temp=temp, finetune=True, record_name='teacher_FineTune')
        KD_system.log_print('*' * 10 + ' Re-initialize student model' + '*' * 10)
        KD_system.re_initialize_student()
        KD_system.log_print('*' * 10 + ' Fine-tuning student with KD, temp = {}'.format(temp) + '*' * 10)
        # for _ in range(7):
        KD_system.train(model=KD_system.student, is_distill=True, temp=temp, finetune=True, record_name='student_FineTune_KD')


        if the_args.use_assistant or the_args.complete:
            KD_system.log_print('*' * 10 + ' Fine-tuning assistant with KD, temp = {}'.format(temp) + '*' * 10)
            KD_system.train(model=KD_system.assistant, is_distill=True, temp=temp, finetune=True, record_name='assistant_FineTune_KD')
            KD_system.log_print('*' * 10 + ' Re-initialize student model' + '*' * 10)
            KD_system.re_initialize_student()
            KD_system.train(model=KD_system.student, teachers=[KD_system.assistant], is_distill=True, temp=temp, finetune=True, record_name='student_FineTune_KD_assistant')

            if the_args.use_both_teachers or the_args.complete:
                KD_system.log_print('*' * 10 + ' Re-initialize student model' + '*' * 10)
                KD_system.log_print('*' * 10 + ' Fine-tuning student with KD using both teachers, temp = {}'.format(temp) + '*' * 10)
                KD_system.re_initialize_student()
                KD_system.train(model=KD_system.student, teachers=[KD_system.teacher, KD_system.assistant], is_distill=True, temp=temp,
                                finetune=True, record_name='student_FineTune_KD_both_teacher')

    if the_args.student_temp_values or the_args.assistant_temp_values:
        KD_system.log_print('*' * 10 + ' Fine-tuning teacher model, temp='.format(temp) + '*' * 10)
        KD_system.train(model=KD_system.teacher, temp=temp, finetune=True, record_name='teacher_FineTune')
        for temp in [1, 2, 4, 16, 32, 64]:
            if the_args.student_temp_values:
                KD_system.re_initialize_student()
                KD_system.log_print('*' * 10 + ' Fine-tuning student with KD, temp = {}'.format(temp) + '*' * 10)
                KD_system.train(model=KD_system.student, is_distill=True, teachers=[KD_system.teacher], temp=temp,
                                finetune=True, record_name='student_KD_temp_{}'.format(temp))

            if the_args.assistant_temp_values:
                KD_system.re_initialize_assistant()
                KD_system.log_print('*' * 10 + ' Fine-tuning assistant with KD, temp = {}'.format(temp) + '*' * 10)
                KD_system.train(model=KD_system.assistant, is_distill=True, teachers=[KD_system.teacher], temp=temp,
                                finetune=True, record_name='assistant_KD_temp_{}'.format(temp))

    KD_system.save_list_to_txt('result.txt')


