import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from model.AlexNet import AlexNet
from model.VGG_16_1D import VGG_16_1D
from model.DAConv_VGG_16_1D import DAConv_VGG_16_1D
from model.ResNet_50_1D import ResNet_50_1D
from model.DAConv_ResNet_50_1D import DAConv_ResNet_50_1D
from model.CVCNN import CVCNN
from model.AFFNet import AFFNet
from config import cfg
import pickle
import torch.nn.functional as F
from utils import read_sample_from_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = {'AlexNet': AlexNet,
         'VGG': VGG_16_1D,
         'VGG-DAConv': DAConv_VGG_16_1D,
         'ResNet': ResNet_50_1D,
         'ResNet-DAConv': DAConv_ResNet_50_1D,
         'CVCNN': CVCNN,
         'AFFNet': AFFNet}
with open('{}mean_std.pkl'.format(cfg['train_data_dir']), 'rb') as f:
    mean_std = pickle.load(f)
mean, std = mean_std[0], mean_std[1]

class SEI():
    def __init__(self, config):
        self.cfg = config
        self.sample_len = config['sample_len']
        self.sample_overlap = config['sample_overlap']
        self.load_model()

    def load_model(self):
        self.net = model[self.cfg['model']](self.cfg).to(device)
        checkpoint = torch.load(self.cfg['checkpoint_path'] + self.cfg['model'] + '/checkpoint_{:02d}.pth'.format(self.cfg['n_epoch']))
        self.net.load_state_dict(checkpoint)

    def infer(self, IQs):
        IQs = torch.from_numpy(IQs)
        IQs = IQs.unsqueeze(0).to(device)
        out = self.net(IQs)
        out = F.softmax(out, dim=-1).cpu().detach().numpy()
        prob = out.max()
        index = out.argmax()
        return index, prob


if __name__ == '__main__':
    sei = SEI(cfg)
    samples = read_sample_from_file('data/static_channel/0.dat', cfg['sample_len'], 10)
    for sample in samples:
        index, prob = sei.infer(sample)
        print('index:{}, prob:{}'.format(index, prob))






