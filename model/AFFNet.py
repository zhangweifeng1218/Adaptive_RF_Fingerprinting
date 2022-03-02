import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init
from config import cfg

class DAConv(nn.Module):
    def __init__(self, sample_len, in_channels, out_channels, kernel_size, stride, padding, spatial_att_enable=True, channel_att_enable=True, residual_enable=True):
        super(DAConv, self).__init__()
        self.sample_len = sample_len
        self.out_len = int((sample_len+2*padding-(kernel_size-1)-1)/stride +1)
        self.spatial_att_enable = spatial_att_enable
        self.channel_att_enable = channel_att_enable
        self.residual_enable = residual_enable
        self.conv_branch = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding)
        if spatial_att_enable:
            self.spatial_att_conv = nn.Conv1d(in_channels, 1, kernel_size, 2)
            self.spatial_att_pool = nn.MaxPool1d(kernel_size, 2)
            self.spatial_att_deconv = nn.ConvTranspose1d(1, 1, kernel_size, 2)
            self.upsample = nn.Upsample(size=(self.out_len), mode='linear')
        if channel_att_enable:
            self.channel_att_fc1 = nn.Linear(sample_len, out_channels)
            self.channel_att_fc2 = nn.Linear(in_channels, 1)



    def forward(self, x):
        conv_out = self.conv_branch(x)
        if self.spatial_att_enable:
            satt_map = F.relu(self.spatial_att_conv(x))
            satt_map = self.spatial_att_pool(satt_map)
            satt_map = F.relu(self.spatial_att_deconv(satt_map))
            satt_map = self.upsample(satt_map)
            satt_map = torch.sigmoid(satt_map)
            if self.residual_enable:
                conv_out = conv_out + conv_out * satt_map
            else:
                conv_out = conv_out * satt_map.unsqueeze(1)
        if self.channel_att_enable:
            catt_map = torch.tanh(self.channel_att_fc1(x))
            catt_map = torch.transpose(catt_map, 1, 2)
            catt_map = self.channel_att_fc2(catt_map)
            catt_map = torch.sigmoid(catt_map)
            catt_map = torch.sum(catt_map, dim=-1, keepdim=True)
            if self.residual_enable:
                conv_out = conv_out + conv_out * catt_map
            else:
                conv_out = conv_out * catt_map
        return conv_out

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.samp_len = cfg['sample_len']
        self.conv1 = DAConv(self.samp_len, 2, 64, 11, 4, 0)
        self.samp_len = int((self.samp_len - 11) / 4 + 1)
        self.pool = nn.MaxPool1d(3, 2)
        self.samp_len = int((self.samp_len - 3) / 2 + 1)
        self.conv2 = DAConv(self.samp_len, 64, 128, 5, 1, 2)
        self.samp_len = int((self.samp_len - 5 + 4) / 1 + 1)
        self.samp_len = int((self.samp_len - 3) / 2 + 1)
        self.conv3 = DAConv(self.samp_len, 128, 128, 3, 1, 1)
        self.samp_len = int((self.samp_len - 1) / 1 + 1)
        self.conv4 = DAConv(self.samp_len, 128, 64, 3, 1, 1)

    def forward(self, x):
        # [N, 2, samp_len]
        x = F.relu(self.conv1(x))  # [512, 96, 126]
        x = self.pool(x)  # [512, 96, 62]
        x = F.relu(self.conv2(x))  # [512, 128, 62]
        x = self.pool(x)  # [512, 128, 30]
        x = F.relu(self.conv3(x))  # [512, 128, 30]
        x = F.relu(self.conv4(x))  # [512, 64, 30]
        x = self.pool(x)  # [512, 64, 14]
        return x


class Classifier(nn.Module):
    def __init__(self, cfg=None):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(896, 128)  # nn.Linear(384, 128)#
        self.fc2 = nn.Linear(128, cfg['n_classes'])

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_feature(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class AFFNet(nn.Module):
    '''简化版的alexnet,效果更好'''
    def __init__(self, cfg):
        super(AFFNet, self).__init__()
        self.encoder = Encoder(cfg)
        self.classifier = Classifier(cfg)

    def forward(self, x):
        feature = self.encoder(x)
        predicted = self.classifier(feature)
        return predicted

    def get_feature(self, x):
        feature = self.encoder(x)
        return self.classifier.get_feature(feature)




if __name__ == '__main__':
    from utils import count_parameters
    x = torch.randn(size=(1, 2, 512))
    net = AFFNet(cfg)
    y = net(x)
    print(y.shape)
    print(count_parameters(net))