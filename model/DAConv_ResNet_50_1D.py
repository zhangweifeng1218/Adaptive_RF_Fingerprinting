import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init
from model.AFFNet import DAConv

class DAConv1(nn.Module):
    def __init__(self, sample_len, in_planes, places, kernel_size=7, stride=2, padding=3):
        super(DAConv1, self).__init__()
        self.daconv = DAConv(sample_len=sample_len, in_channels=in_planes,out_channels=places,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm1d(places)
        self.relu = nn.ReLU(inplace=True)
        self.out_len = self.daconv.out_len

    def forward(self, x):
        x = self.daconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv1(nn.Module):
    def __init__(self, sample_len, in_planes, places, kernel_size=7, stride=2, padding=3):
        super(Conv1, self).__init__()
        self.daconv = nn.Conv1d(in_channels=in_planes,out_channels=places,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm1d(places)
        self.relu = nn.ReLU(inplace=True)
        self.out_len = int((sample_len + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

    def forward(self, x):
        x = self.daconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, sample_len, in_channel, filters, s=1):
        super(BasicBlock, self).__init__()
        F1, F2 = filters
        self.s = s
        self.conv1 = DAConv1(sample_len, in_channel, F1, kernel_size=3, stride=s, padding=True)
        self.conv2 = DAConv1(self.conv1.out_len, F1, F2, kernel_size=3, stride=1, padding=True)
        self.out_len = self.conv2.out_len
        if s > 1:
            self.shortcut_1 = nn.Conv1d(in_channel, F2, 1, stride=s, padding=0, bias=False)
            self.batch_1 = nn.BatchNorm1d(F2)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        if self.s > 1:
            X_shortcut = self.shortcut_1(X)
            X_shortcut = self.batch_1(X_shortcut)
        else:
            X_shortcut = X
        X = self.conv1(X)
        X = self.conv2(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class ConvBlock(nn.Module):
    def __init__(self, sample_len, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.conv1 = Conv1(sample_len, in_channel, F1, kernel_size=1, stride=s, padding=0)
        self.conv2 = DAConv1(self.conv1.out_len, F1, F2, kernel_size=f, stride=1, padding=True)
        self.conv3 = Conv1(self.conv2.out_len, F2, F3, kernel_size=1, stride=1, padding=0)
        self.out_len = self.conv3.out_len
        self.shortcut_1 = nn.Conv1d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm1d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, sample_len, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.conv1 = Conv1(sample_len, in_channel, F1, kernel_size=1, stride=1, padding=0)
        self.conv2 = DAConv1(self.conv1.out_len, F1, F2, kernel_size=f, stride=1, padding=True)
        self.conv3 = Conv1(self.conv2.out_len, F2, F3, kernel_size=1, stride=1, padding=0)
        self.out_len = self.conv3.out_len
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class DAConv_ResNet_34_1D(nn.Module):
    def __init__(self, cfg=None):
        super(DAConv_ResNet_34_1D, self).__init__()
        sample_len = cfg['sample_len']
        n_class = cfg['n_classes']
        self.stage1_conv = Conv1(sample_len, 2, 64, 7, stride=2, padding=3)
        self.stage1_maxpool = nn.MaxPool1d(3, 2, padding=1)
        self.stage1_out_len = int((self.stage1_conv.out_len+2*1-(3-1)-1)/2 + 1)

        self.stage2_1 = BasicBlock(self.stage1_out_len, 64, filters=[64, 64], s=1)
        self.stage2_2 = BasicBlock(self.stage1_out_len, 64, filters=[64, 64], s=1)
        self.stage2_3 = BasicBlock(self.stage1_out_len, 64, filters=[64, 64], s=1)
        self.stage2_out_len = self.stage2_3.out_len

        self.stage3_1 = BasicBlock(self.stage2_out_len, 64, filters=[128, 128], s=2)
        self.stage3_2 = BasicBlock(self.stage3_1.out_len, 128, filters=[128, 128], s=1)
        self.stage3_3 = BasicBlock(self.stage3_2.out_len, 128, filters=[128, 128], s=1)
        self.stage3_4 = BasicBlock(self.stage3_3.out_len, 128, filters=[128, 128], s=1)
        self.stage3_out_len = self.stage3_4.out_len

        self.stage4_1 = BasicBlock(self.stage3_out_len, 128, filters=[256, 256], s=2)
        self.stage4_2 = BasicBlock(self.stage4_1.out_len, 256, filters=[256, 256], s=1)
        self.stage4_3 = BasicBlock(self.stage4_2.out_len, 256, filters=[256, 256], s=1)
        self.stage4_4 = BasicBlock(self.stage4_3.out_len, 256, filters=[256, 256], s=1)
        self.stage4_5 = BasicBlock(self.stage4_4.out_len, 256, filters=[256, 256], s=1)
        self.stage4_6 = BasicBlock(self.stage4_5.out_len, 256, filters=[256, 256], s=1)
        self.stage4_out_len = self.stage4_6.out_len

        self.stage5_1 = BasicBlock(self.stage4_out_len, 256, filters=[512, 512], s=2)
        self.stage5_2 = BasicBlock(self.stage5_1.out_len, 512, filters=[512, 512], s=1)
        self.stage5_3 = BasicBlock(self.stage5_2.out_len, 512, filters=[512, 512], s=1)

        self.pool = nn.AdaptiveAvgPool2d((512, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, n_class)
        )

    def forward(self, X):
        out = self.stage1_conv(X)
        out = self.stage1_maxpool(out)

        out = self.stage2_1(out)
        out = self.stage2_2(out)
        out = self.stage2_3(out)

        out = self.stage3_1(out)
        out = self.stage3_2(out)
        out = self.stage3_3(out)
        out = self.stage3_4(out)

        out = self.stage4_1(out)
        out = self.stage4_2(out)
        out = self.stage4_3(out)
        out = self.stage4_4(out)
        out = self.stage4_5(out)
        out = self.stage4_6(out)

        out = self.stage5_1(out)
        out = self.stage5_2(out)
        out = self.stage5_3(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class DAConv_ResNet_50_1D(nn.Module):
    def __init__(self, cfg=None):
        super(DAConv_ResNet_50_1D, self).__init__()
        sample_len = cfg['sample_len']
        n_class = cfg['n_classes']
        self.stage1_conv = Conv1(sample_len, 2, 64, 7, stride=2, padding=3)
        self.stage1_maxpool = nn.MaxPool1d(3, 2, padding=1)
        self.stage1_out_len = int((self.stage1_conv.out_len+2*1-(3-1)-1)/2 + 1)

        self.stage2_CB = ConvBlock(self.stage1_out_len, 64, f=3, filters=[64, 64, 256], s=1)
        self.stage2_IB1 = IndentityBlock(self.stage2_CB.out_len, 256, 3, [64, 64, 256])
        self.stage2_IB2 = IndentityBlock(self.stage2_IB1.out_len, 256, 3, [64, 64, 256])
        self.stage2_out_len = self.stage2_IB2.out_len

        self.stage3_CB = ConvBlock(self.stage2_out_len, 256, f=3, filters=[128, 128, 512], s=2)
        self.stage3_IB1 = IndentityBlock(self.stage3_CB.out_len, 512, 3, [128, 128, 512])
        self.stage3_IB2 = IndentityBlock(self.stage3_IB1.out_len, 512, 3, [128, 128, 512])
        self.stage3_IB3 = IndentityBlock(self.stage3_IB2.out_len, 512, 3, [128, 128, 512])
        self.stage3_out_len = self.stage3_IB3.out_len

        self.stage4_CB = ConvBlock(self.stage3_out_len, 512, f=3, filters=[256, 256, 1024], s=2)
        self.stage4_IB1 = IndentityBlock(self.stage4_CB.out_len, 1024, 3, [256, 256, 1024])
        self.stage4_IB2 = IndentityBlock(self.stage4_IB1.out_len, 1024, 3, [256, 256, 1024])
        self.stage4_IB3 = IndentityBlock(self.stage4_IB2.out_len, 1024, 3, [256, 256, 1024])
        self.stage4_IB4 = IndentityBlock(self.stage4_IB3.out_len, 1024, 3, [256, 256, 1024])
        self.stage4_IB5 = IndentityBlock(self.stage4_IB4.out_len, 1024, 3, [256, 256, 1024])
        self.stage4_out_len = self.stage4_IB5.out_len

        self.stage5_CB = ConvBlock(self.stage4_out_len, 1024, f=3, filters=[512, 512, 2048], s=2)
        self.stage5_IB1 = IndentityBlock(self.stage5_CB.out_len, 2048, 3, [512, 512, 2048])
        self.stage5_IB2 = IndentityBlock(self.stage5_IB1.out_len, 2048, 3, [512, 512, 2048])

        self.pool = nn.AdaptiveAvgPool2d((512, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, n_class)
        )

    def forward(self, X):
        out = self.stage1_conv(X)
        out = self.stage1_maxpool(out)

        out = self.stage2_CB(out)
        out = self.stage2_IB1(out)
        out = self.stage2_IB2(out)

        out = self.stage3_CB(out)
        out = self.stage3_IB1(out)
        out = self.stage3_IB2(out)
        out = self.stage3_IB3(out)

        out = self.stage4_CB(out)
        out = self.stage4_IB1(out)
        out = self.stage4_IB2(out)
        out = self.stage4_IB3(out)
        out = self.stage4_IB4(out)
        out = self.stage4_IB5(out)

        out = self.stage5_CB(out)
        out = self.stage5_IB1(out)
        out = self.stage5_IB2(out)

        out = self.pool(out)
        out = out.view(out.size(0), 512)
        out = self.fc(out)
        return out

if __name__=='__main__':
    from config import cfg
    model = DAConv_ResNet_34_1D(cfg)
    print(model)

    input = torch.randn(1, 2, 512)
    out = model(input)
    print(out.shape)
    from utils import count_parameters
    print(count_parameters(model))