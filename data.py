import os
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset
import h5py
import pickle
from config import cfg

class SEIDataset(Dataset):
    def __init__(self, data_file, split='train'):
        super(SEIDataset, self).__init__()
        self.data_file = data_file
        self.split = split
        self.hf = None

    def __len__(self):
        if self.hf is not None: self.hf.close()
        self.hf = h5py.File(self.data_file, 'r')
        length = len(self.hf['{}_label'.format(self.split)])
        self.hf.close()
        return length

    def __getitem__(self, index):
        if self.hf is not None: self.hf.close()
        self.hf = h5py.File(self.data_file, 'r')
        data = self.hf['{}_data'.format(self.split)][index]
        label = self.hf['{}_label'.format(self.split)][index]
        self.hf.close()
        return torch.from_numpy(data), torch.from_numpy(label).long()

class SSLDataset(Dataset):
    '''用于自监督学习'''
    def __init__(self, data_file, split='train'):
        super(SSLDataset, self).__init__()
        self.data_file = data_file
        self.split = split
        self.hf = None
        self.classes = [0, 1, 2, 3, 4]
        self.targets = []
        with h5py.File(self.data_file, 'r') as hf:
            self.targets = [int(target) for target in hf['{}_label'.format(self.split)]]
        with open('{}{}_label_index.pkl'.format(cfg['train_data_dir'], self.split), 'rb') as f:
            self.label_index_dict = pickle.load(f)

    def __len__(self):
        if self.hf is not None: self.hf.close()
        self.hf = h5py.File(self.data_file, 'r')
        length = len(self.hf['{}_label'.format(self.split)])
        self.hf.close()
        return length

    def __getitem__(self, index):
        if self.hf is not None: self.hf.close()
        self.hf = h5py.File(self.data_file, 'r')
        data = self.hf['{}_data'.format(self.split)][index]
        label = self.hf['{}_label'.format(self.split)][index]
        label_indexes = self.label_index_dict[label[0]]
        paired_data_index = np.random.randint(len(label_indexes))
        paired_data = self.hf['{}_data'.format(self.split)][paired_data_index]
        self.hf.close()
        return torch.from_numpy(data), torch.from_numpy(paired_data), torch.from_numpy(label).long()





