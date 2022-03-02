import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from config import cfg
import pickle
import matplotlib.pyplot as pl
from sklearn.metrics import confusion_matrix

def draw(x, y, legend_labels, x_label, y_label, fig_name):
    color = ['red', 'blue', 'dimgray', 'coral', 'cyan', 'black']
    font = {'family': 'Times New Roman',
            'size': 30}
    tick_size = 23
    figsize = 10, 7
    figure, ax = pl.subplots(figsize=figsize)
    handles = []
    for i in range(len(x)):
        tmp_x = x[i]
        tmp_y = y[i]
        ln, = pl.plot(tmp_x, tmp_y, color=color[i])
        handles.append(ln)
    pl.legend(handles=handles, labels=legend_labels, prop=font)
    pl.tick_params(labelsize=tick_size)
    pl.xlabel(x_label, fontdict=font)
    pl.ylabel(y_label, fontdict=font)
    pl.grid()
    pl.savefig(fig_name)
    pl.show()

def show_loss_acc_curve(data_file, fig_dir):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    train_loss = data['train_loss']
    train_moving_loss = data['train_moving_loss']
    test_loss = data['test_loss']
    test_moving_loss = data['test_moving_loss']
    train_acc = data['train_acc']
    train_moving_acc = data['train_moving_acc']
    test_acc = data['test_acc']
    test_moving_acc = data['test_moving_acc']
    loss_iters = range(len(train_loss))
    acc_iters = range(len(train_acc))
    draw([loss_iters, loss_iters], [train_loss, train_moving_loss], ['loss', 'moving loss'], 'iter', 'loss', '{}train_loss.pdf'.format(fig_dir))
    draw([acc_iters, acc_iters], [train_acc, train_moving_acc], ['acc', 'moving acc'], 'iter', 'acc', '{}train_acc.pdf'.format(fig_dir))
    loss_iters = range(len(test_loss))
    acc_iters = range(len(test_acc))
    draw([loss_iters, loss_iters], [test_loss, test_moving_loss], ['loss', 'moving loss'], 'iter', 'loss',
         '{}test_loss.pdf'.format(fig_dir))
    draw([acc_iters, acc_iters], [test_acc, test_moving_acc], ['acc', 'moving acc'], 'iter', 'acc', '{}test_acc.pdf'.format(fig_dir))


def plot_confusion_matrix(classes, cm, savename, title='Confusion Matrix'):
    pl.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            pl.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    pl.imshow(cm, interpolation='nearest', cmap=pl.cm.binary)
    pl.title(title)
    pl.colorbar()
    xlocations = np.array(range(len(classes)))
    pl.xticks(xlocations, classes, rotation=90)
    pl.yticks(xlocations, classes)
    pl.ylabel('Actual label')
    pl.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    pl.gca().set_xticks(tick_marks, minor=True)
    pl.gca().set_yticks(tick_marks, minor=True)
    pl.gca().xaxis.set_ticks_position('none')
    pl.gca().yaxis.set_ticks_position('none')
    pl.grid(True, which='minor', linestyle='-')
    pl.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    pl.savefig(savename)
    pl.show()

def get_mean_std(dataset, ratio=0.01):
    """从数据集中安指定比例（ratio）随机采样获得样本，计算这些样本的mean和std
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=True)
    train = iter(dataloader).next()[0]   # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2))
    std = np.std(train.numpy(), axis=(0, 2))
    return mean, std

def read_sample_from_file(file_name, IQ_per_sample, sample_num, overlap=0, offset=0):
    '''
    从文件读取IQ信号样本
    :param file_name: 文件名
    :param IQ_per_sample: 每个样本中的采样点数
    :param sample_num: 读取样本数
    :return:
    '''
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'r') as f:
        f.seek(offset, 0)
        data = np.fromfile(f, dtype=np.float32)
        I = data[0:data.size:2]
        Q = data[1:data.size:2]
        data_len = min(I.size, Q.size)
        I = I[0:data_len]
        Q = Q[0:data_len]
        if data_len >= IQ_per_sample * sample_num:
            IQs = []
            start_index = 0
            for k in range(sample_num):
                tmp_I = I[start_index:start_index+IQ_per_sample]
                tmp_Q = Q[start_index:start_index+IQ_per_sample]
                IQ = np.concatenate(([tmp_I], [tmp_Q]), axis=0)
                IQs.append(IQ)
                start_index += IQ_per_sample - overlap
            return IQs
        else:
            return None

def read_file(file_name, bytes, begining=0):
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'r') as f:
        f.seek(0, 2)
        file_length = f.tell()
        if file_length >= bytes:
            if begining == 2:
                f.seek(-bytes, 2)
            else:
                f.seek(0, 0)
            data = np.fromfile(f, dtype=np.float32)
            I = data[0:data.size:2]
            Q = data[1:data.size:2]
            data_len = min(I.size, Q.size)
            I = I[0:data_len]
            Q = Q[0:data_len]
            IQ = np.concatenate(([I], [Q]), axis=0)
            return IQ
        else:
            return None


def bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1) # multiply by number of QAs
    return loss

def lr_schedule_func_builder(warmup=2, warmup_step=2, warmup_factor=0.2, keep_steps=5, decay_step=10, decay_ratio = 0.5):
    def func(epoch):
        alpha = float(warmup) / float(warmup_step)
        warmed_ratio = warmup_factor * (1. - alpha) + alpha
        if epoch <= warmup:
            alpha = float(epoch) / float(warmup_step)
            return warmup_factor * (1. - alpha) + alpha
        else:
            if epoch < warmup+keep_steps:
                return warmed_ratio
            else:
                idx = int((epoch-keep_steps)/decay_step)
                return pow(decay_ratio, idx) * warmed_ratio
    return func

def count_parameters(model):
    return sum(p.nelement() for p in model.parameters() if p.requires_grad)

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.items():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        #print(msg)

    