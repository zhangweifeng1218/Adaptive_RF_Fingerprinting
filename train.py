import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
from data import SEIDataset
from model.AlexNet import AlexNet
from model.VGG_16_1D import VGG_16_1D
from model.DAConv_VGG_16_1D import DAConv_VGG_16_1D
from model.ResNet_50_1D import ResNet_50_1D
from model.DAConv_ResNet_50_1D import DAConv_ResNet_50_1D
from model.CVCNN import CVCNN
from model.AFFNet import AFFNet
from utils import count_parameters, lr_schedule_func_builder, Logger, show_loss_acc_curve
from config import cfg
import pickle
import time


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


def normalize(x, m, s):
    batch_size = x.size(0)
    length = x.size(2)
    m = torch.from_numpy(m)
    m = m.unsqueeze(0).unsqueeze(2)
    m = m.repeat(batch_size, 1, length)
    s = torch.from_numpy(s)
    s = s.unsqueeze(0).unsqueeze(2)
    x = (x-m)/s
    return x


def bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


def compute_score_with_logits(logits, labels):
    with torch.no_grad():
        logits = torch.max(logits, 1)[1] # argmax
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)
        return scores


def train(epoch):
    dataset = iter(train_loader)
    pbar = tqdm(dataset)
    epoch_score = 0.0
    epoch_loss = 0.0
    moving_score = 0
    moving_loss = 0
    n_samples = 0
    net.train(True)
    start_time = time.time()
    for IQ_slice, label in pbar:
        IQ_slice = normalize(IQ_slice, mean, std)
        IQ_slice, label = (
            IQ_slice.to(device),
            label.to(device),
        )
        n_samples += label.size(0)

        net.zero_grad()
        output = net(IQ_slice)


        target = torch.zeros_like(output).scatter_(1, label.view(-1, 1), 1)
        loss = criterion(output, target)
        loss.backward()

        clip_grad_norm_(net.parameters(), 0.25)
        optimizer.step()

        batch_score = compute_score_with_logits(output, target).sum()
        epoch_loss += float(loss.data.item()) * target.size(0)
        epoch_score += float(batch_score)
        moving_loss = epoch_loss / n_samples
        moving_score = epoch_score / n_samples
        loss_acc['train_loss'].append(float(loss.data.item()))
        loss_acc['train_acc'].append(float(batch_score)/label.size(0))
        loss_acc['train_moving_loss'].append(moving_loss)
        loss_acc['train_moving_acc'].append(moving_score)

        pbar.set_description(
            'Train Epoch: {}; Loss: {:.6f}; Acc: {:.6f}'.format(epoch + 1, moving_loss, moving_score))

    end_time = time.time()
    print(end_time-start_time)

    logger.write('Epoch: {:2d}: Train Loss: {:.6f}; Train Acc: {:.4f}'.format(epoch+1, moving_loss, moving_score))


def test(epoch):
    dataset = iter(test_loader)
    pbar = tqdm(dataset)
    epoch_score = 0.0
    epoch_loss = 0.0
    moving_score = 0
    moving_loss = 0
    n_samples = 0
    net.eval()
    with torch.no_grad():
        for IQ_slice, label in pbar:
            IQ_slice, label = (
                IQ_slice.to(device),
                label.to(device),
            )
            n_samples += label.size(0)

            output = net(IQ_slice)


            target = torch.zeros_like(output).scatter_(1, label.view(-1, 1), 1)
            loss = criterion(output, target)

            batch_score = compute_score_with_logits(output, target).sum()
            epoch_loss += float(loss.data.item()) * target.size(0)
            epoch_score += float(batch_score)
            moving_loss = epoch_loss / n_samples
            moving_score = epoch_score / n_samples
            loss_acc['test_loss'].append(float(loss.data.item()))
            loss_acc['test_acc'].append(float(batch_score)/label.size(0))
            loss_acc['test_moving_loss'].append(moving_loss)
            loss_acc['test_moving_acc'].append(moving_score)

            pbar.set_description(
                'Val Epoch: {}; Loss: {:.6f}; Acc: {:.6f}'.format(epoch + 1, moving_loss, moving_score))

    logger.write('Val: {:2d}: Loss: {:.6f}; Acc: {:.4f}'.format(epoch+1, moving_loss, moving_score))


if __name__ == '__main__':
    np.random.seed(101)  # numpy
    random.seed(101)  # random and transforms
    torch.manual_seed(101)
    #torch.cuda.manual_seed(101)

    print('Loading data...')
    train_dataset = SEIDataset(data_file=cfg['h5_file'], split='train')
    test_dataset = SEIDataset(data_file=cfg['h5_file'], split='test')
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True)

    print('Creating Model...')
    net = model[cfg['model']](cfg).to(device)
    n_params = count_parameters(net)
    print("model: {:,}  parameters".format(n_params))

    ssl_checkpoint = '{}/ssl_checkpoint_{}.pth'.format(cfg['checkpoint_path'] + cfg['model'], str(cfg['ssl_n_epoch']).zfill(2))
    if os.path.exists(ssl_checkpoint):
        net.encoder.load_state_dict(torch.load(ssl_checkpoint))

    criterion = bce_with_logits
    optimizer = optim.Adam(net.parameters(), lr=cfg['lr'])
    #optimizer = optim.Adam([
    #    {'params': net.encoder.parameters(), 'lr': cfg['ssl_finetune_lr']},
    #    {'params': net.classifier.parameters()}
    #], lr=cfg['lr'])
    sched = LambdaLR(optimizer, lr_lambda=lr_schedule_func_builder())

    checkpoint_path = cfg['checkpoint_path'] + cfg['model']
    if os.path.exists(checkpoint_path) is False:
        os.mkdir(checkpoint_path)

    logger = Logger(os.path.join(checkpoint_path, "log.txt"))
    for k, v in cfg.items():
        logger.write(k+': {}'.format(v))

    loss_acc = {
        'train_loss': [],
        'train_acc': [],
        'train_moving_loss': [],
        'train_moving_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_moving_loss': [],
        'test_moving_acc': []}

    print('Starting train...')
    for epoch in range(cfg['n_epoch']):
        print('\n lr={}'.format(optimizer.state_dict()["param_groups"][0]["lr"]))
        train(epoch)
        test(epoch)
        sched.step()

        if (epoch+1) % 5 == 0:
            with open('{}/checkpoint_{}.pth'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(net.state_dict(), f)
            with open('{}/optim_checkpoint_{}.pth'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(optimizer.state_dict(), f)

    with open('{}/loss_acc.pkl'.format(checkpoint_path), 'wb') as f:
        pickle.dump(loss_acc, f)

    show_loss_acc_curve('{}/loss_acc.pkl'.format(checkpoint_path), checkpoint_path)


