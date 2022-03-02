import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init
from model.fc import  FCNet


def ConvBN(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding),
       # nn.BatchNorm1d(out_channels)
    )



class AlexNet(nn.Module):
    '''简化版的alexnet,效果更好'''
    def __init__(self, cfg):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, 11, 4, 0)
        self.pool = nn.MaxPool1d(3, 2)
        self.conv2 = nn.Conv1d(64, 128, 5, 1, 2)
        self.conv3 = nn.Conv1d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv1d(128, 64, 3, 1, 1)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(896, 128) # nn.Linear(384, 128)#
        self.fc2 = nn.Linear(128, cfg['n_classes'])

    def forward(self, x):
        # [N, 2, samp_len]
        x = F.relu(self.conv1(x))     # [512, 96, 126]
        x = self.pool(x)              # [512, 96, 62]
        x = F.relu(self.conv2(x))     # [512, 128, 62]
        x = self.pool(x)              # [512, 128, 30]
        x = F.relu(self.conv3(x))     # [512, 128, 30]
        x = F.relu(self.conv4(x))     # [512, 64, 30]
        x = self.pool(x)              # [512, 64, 14]
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    from config import cfg
    x = torch.randn(size=(1, 2, 512))
    net = AlexNet(cfg)
    y = net(x)
    print(y.shape)
    from utils import count_parameters
    print(count_parameters(net))