import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        # nn.BatchNorm1d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )

class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv1d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm1d(F1),
            nn.ReLU(True),
            nn.Conv1d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm1d(F2),
            nn.ReLU(True),
            nn.Conv1d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(F3),
        )
        self.shortcut_1 = nn.Conv1d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm1d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv1d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(F1),
            nn.ReLU(True),
            nn.Conv1d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm1d(F2),
            nn.ReLU(True),
            nn.Conv1d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResNet_50_1D(nn.Module):
    def __init__(self, cfg=None):
        super(ResNet_50_1D, self).__init__()
        n_class = cfg['n_classes']
        self.stage1 = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
            IndentityBlock(2048, 3, [512, 512, 2048]),
            IndentityBlock(2048, 3, [512, 512, 2048]),
        )
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(9225, n_class)
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        out = out.view(out.size(0), 9225)
        out = self.fc(out)
        return out

if __name__=='__main__':
    from config import cfg
    model = ResNet_50_1D(cfg)
    print(model)

    input = torch.randn(1, 2, 512)
    out = model(input)
    print(out.shape)
    from utils import count_parameters
    print(count_parameters(model))