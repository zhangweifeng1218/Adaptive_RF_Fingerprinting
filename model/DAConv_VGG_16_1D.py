import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init
from model.fc import  FCNet
from model.AFFNet import DAConv


class Conv1(nn.Module):
    def __init__(self, sample_len, in_planes, places, kernel_size=7, stride=2, padding=3):
        super(Conv1, self).__init__()
        self.daconv = DAConv(sample_len=sample_len, in_channels=in_planes,out_channels=places,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm1d(places)
        self.relu = nn.ReLU(inplace=True)
        self.out_len = self.daconv.out_len

    def forward(self, x):
        x = self.daconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DAConv_VGG(nn.Module):
    def __init__(self, samp_num=512, num_classes=5, block_nums=None):
        super(DAConv_VGG, self).__init__()
        self.len = samp_num
        self.stage1 = self._make_layers(sample_len=self.len, in_channels=2, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(sample_len=self.len, in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(sample_len=self.len, in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(sample_len=self.len, in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(sample_len=self.len, in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=8192 ,out_features=1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_classes)
        )

        self._init_params()

    def _make_layers(self, sample_len, in_channels, out_channels, block_num):
        layers = []
        conv = Conv1(sample_len, in_channels, out_channels, kernel_size=3,stride=1,padding=1)
        layers.append(conv)
        for i in range(1,block_num):
            conv = Conv1(conv.out_len, out_channels,out_channels, kernel_size=3,stride=1,padding=1)
            layers.append(conv)
        layers.append(nn.MaxPool1d(kernel_size=2,stride=2, ceil_mode=False))
        self.len = int((conv.out_len - 2)/2 + 1)
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out


def DAConv_VGG_16_1D(cfg=None):
    block_nums = [2, 2, 3, 3, 3]
    model = DAConv_VGG(cfg['sample_len'], cfg['n_classes'], block_nums)
    return model


if __name__ == '__main__':
    from config import cfg
    x = torch.randn(size=(1, 2, 512))
    net = DAConv_VGG_16_1D(cfg)
    y = net(x)
    print(y.shape)
    from utils import count_parameters
    print(count_parameters(net))
