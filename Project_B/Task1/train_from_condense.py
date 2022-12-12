import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=20, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--distilled_set_name', type=str, default='result/res_DC_CIFAR10_ConvNet_10ipc.pt', help='distance metric')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    distilled_set_name = args.distilled_set_name
    dataset = torch.load(distilled_set_name)
    syn_data = dataset['data']
    syn_data = TensorDataset(syn_data[0][0], syn_data[0][1])

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_train = []
        labels_train = []
        indices_class = [[] for c in range(num_classes)]

        images_train = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_train = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_train):
            indices_class[lab].append(i)
        images_train = torch.cat(images_train, dim=0).to(args.device)
        labels_train = torch.tensor(labels_train, dtype=torch.long, device=args.device)

        images_test = [torch.unsqueeze(dst_test[i][0], dim=0) for i in range(len(dst_test))]
        labels_test = [dst_test[i][1] for i in range(len(dst_test))]
        images_test = torch.cat(images_test, dim=0).to(args.device)
        labels_test = torch.tensor(labels_test, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_train[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_train[:, ch]), torch.std(images_train[:, ch])))


        ''' training '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        dataset_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(syn_data, batch_size=args.batch_train, shuffle=True, num_workers=0)
        
        dataset_test = TensorDataset(images_test, labels_test)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_train, shuffle=True, num_workers=0)
                
        acc_train = []
        acc_test  = []
        exposides = []

        for it in range(args.Iteration+1):
            
            loss, acc  = epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)
            acc_train.append(acc)
            loss, acc  = epoch('eval', testloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)
            acc_test.append(acc)
            exposides.append(it)
            print('Epoch' , it+1, 'Accuracy on train condensed dataset', acc_train[it], ' Accuracy on test dataset', acc_test[it])           

        print('%s training end'%get_time())
        fig, ax = plt.subplots()

        ax.plot(exposides, acc_train, label='accuracy on train condensed dataset') 
        ax.plot(exposides, acc_test, label='accuracy on test dataset') 

        ax.set_xlabel('Epochs') #设置x轴名称 x label
        ax.set_ylabel('Accuracy') #设置y轴名称 y label
        ax.set_title('The accuracy results of training and testing') #设置图名为Simple Plot
        ax.legend() #自动检测要在图例中显示的元素，并且显示
        fig.savefig('./train_from_condense.jpg')


if __name__ == '__main__':
    main()


