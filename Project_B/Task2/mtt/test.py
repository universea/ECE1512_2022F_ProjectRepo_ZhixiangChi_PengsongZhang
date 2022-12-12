import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os

feats = torch.load('D:/01_code/mtt-distillation-main/logged_files/CIFAR10/fanciful-field-13/images_best.pt')
# labels = torch.load('D:/01_code/mtt-distillation-main/logged_files/CIFAR100/peach-feather-5/labels_best.pt')
for i in range(feats.shape[0]):
    print(i)
    image_name = "{}.png".format(i)
    path = os.path.join('checkpoins/CIFAR10/result/images_best/', image_name)
    plt.imshow(transforms.ToPILImage()(feats[i]), interpolation='bicubic')
    plt.axis('off')
    plt.savefig(path)
    # plt.show()
