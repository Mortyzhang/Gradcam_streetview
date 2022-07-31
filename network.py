import torch
from torchvision.models import resnet101 #这里可以换模型
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import numpy as np
import os


class Network(nn.Module):
    def __init__(self, n_class):
        super(Network, self).__init__()
        # 搭建网络
        self.net = resnet101(pretrained=True)  # pretrained设置为True，程序会自动下载已经训练好的参数
        # for param in self.net.parameters():  # 这一句可以让model的参数不可训练。注意：它不会影响下面新加的层的权重！！
        #     param.requires_grad = False
        # print(self.net)
        num_features = self.net.fc.in_features  # 512
        self.net.fc = nn.Linear(num_features, n_class)  # 把原来的1000分类的全连接层，转为你要的分类

    def forward(self, x):
        # 这里是取出特征图
        for name, module in self.net._modules.items():
            # print(name, module)
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name is 'layer4':  # 取出layer4层的输出特征。这里可以改成你想要的层
                feature = x
        logits = x
        return feature, logits


def main():
    inputs = torch.ones([10, 3, 64, 64]).cuda()
    model = Network(n_class=2).cuda()
    feature, logits = model(inputs)
    print(feature.shape, logits.shape)


if __name__ == '__main__':
    main()
