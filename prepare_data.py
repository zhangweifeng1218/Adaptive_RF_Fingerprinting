import numpy as np
import os
import h5py
from config import cfg
from utils import get_mean_std
from data import SEIDataset
import pickle

def gen_dataset(data_dir, saved_file_name, sample_len, train_sample_num, test_sample_num, train_test_from_same=True, test_data_dir=None):
    '''
    生成h5数据集文件，h5['train_data'], h5['train_label'], h5['test_data'], h5['test_label']
    :param data_dir: 原始数据目录（文件名表示辐射源名）
    :param saved_file_name: 生成的h5文件
    :param sample_len: 每个样本的采样点数
    :param train_sample_num: 每个辐射源个体的训练样本数
    :param test_sample_num: 每个辐射源个体的测试样本数
    :param train_test_from_same:true：训练和测试样本来自同一文件（static channel），False：来自不同文件（dynamic channel）
    :return:
    '''
    train_label_index_dict = {}
    test_label_index_dict = {}
    if train_test_from_same:
        files = os.listdir(data_dir)
        for file in files:
            if file.split('.')[1] != 'dat':
                continue
            print('processing {} ...'.format(file))
            with open(os.path.join(data_dir, file), 'rb') as f:
                label_index = float(file.split('.')[0])
                data = np.fromfile(f, dtype=np.float32)
                data = data[2560:]
                I = data[0:data.size:2]
                Q = data[1:data.size:2]
                data_len = min(I.size, Q.size)
                I = I[0:data_len]
                Q = Q[0:data_len]
                IQ = np.concatenate(([I], [Q]), axis=0)
                start_index = 0
                samples = []
                labels = []
                while start_index + sample_len < data_len:
                    sample = IQ[:, start_index:start_index+sample_len]
                    start_index += sample_len - cfg['sample_overlap']
                    label = np.array([label_index], dtype=np.float32)
                    samples.append(sample)
                    labels.append(label)
                shuffled_index = list(range(len(samples)))
                np.random.shuffle(shuffled_index)
                train_index = shuffled_index[:train_sample_num]
                test_index = shuffled_index[train_sample_num:train_sample_num+test_sample_num]
                with h5py.File(saved_file_name, 'a') as hf:
                    hf['train_data'].resize(hf['train_data'].shape[0] + len(train_index), axis=0)
                    hf['train_label'].resize(hf['train_label'].shape[0] + len(train_index), axis=0)
                    hf['train_data'][-len(train_index):] = [samples[i] for i in train_index]
                    hf['train_label'][-len(train_index):] = [labels[i] for i in train_index]
                    for i in train_index:
                        if labels[i][0] not in train_label_index_dict.keys():
                            train_label_index_dict[labels[i][0]] = [i]
                        else:
                            train_label_index_dict[labels[i][0]].append(i)
                    hf['test_data'].resize(hf['test_data'].shape[0] + len(test_index), axis=0)
                    hf['test_label'].resize(hf['test_label'].shape[0] + len(test_index), axis=0)
                    hf['test_data'][-len(test_index):] = [samples[i] for i in test_index]
                    hf['test_label'][-len(test_index):] = [labels[i] for i in test_index]
                    for i in test_index:
                        if labels[i][0] not in test_label_index_dict.keys():
                            test_label_index_dict[labels[i][0]] = [i]
                        else:
                            test_label_index_dict[labels[i][0]].append(i)
                print('generating {} train samples, {} test samples'.format(len(train_index), len(test_index)))
    else:
        # train samples
        train_files = os.listdir(data_dir)
        for file in train_files:
            print('processing {} ...'.format(file))
            with open(os.path.join(data_dir, file), 'rb') as f:
                label_index = float(file.split('.')[0])
                data = np.fromfile(f, dtype=np.float32)
                data = data[2560:]
                I = data[0:data.size:2]
                Q = data[1:data.size:2]
                data_len = min(I.size, Q.size)
                I = I[0:data_len]
                Q = Q[0:data_len]
                IQ = np.concatenate(([I], [Q]), axis=0)
                start_index = 0
                samples = []
                labels = []
                while start_index + sample_len < data_len:
                    sample = IQ[:, start_index:start_index + sample_len]
                    start_index += sample_len - cfg['sample_overlap']
                    label = np.array([label_index], dtype=np.float32)
                    samples.append(sample)
                    labels.append(label)
                shuffled_index = list(range(len(samples)))
                np.random.shuffle(shuffled_index)
                train_index = shuffled_index[:train_sample_num]
                with h5py.File(saved_file_name, 'a') as hf:
                    hf['train_data'].resize(hf['train_data'].shape[0] + len(train_index), axis=0)
                    hf['train_label'].resize(hf['train_label'].shape[0] + len(train_index), axis=0)
                    hf['train_data'][-len(train_index):] = [samples[i] for i in train_index]
                    hf['train_label'][-len(train_index):] = [labels[i] for i in train_index]
                    for i in train_index:
                        if labels[i][0] not in train_label_index_dict.keys():
                            train_label_index_dict[labels[i][0]] = [i]
                        else:
                            train_label_index_dict[labels[i][0]].append(i)
                print('generating {} train samples'.format(len(train_index)))
        # test samples
        test_files = os.listdir(test_data_dir)
        for file in test_files:
            print('processing {} ...'.format(file))
            with open(os.path.join(data_dir, file), 'rb') as f:
                label_index = float(file.split('.')[0])
                data = np.fromfile(f, dtype=np.float32)
                data = data[2560:]
                I = data[0:data.size:2]
                Q = data[1:data.size:2]
                data_len = min(I.size, Q.size)
                I = I[0:data_len]
                Q = Q[0:data_len]
                IQ = np.concatenate(([I], [Q]), axis=0)
                start_index = 0
                samples = []
                labels = []
                while start_index + sample_len < data_len:
                    sample = IQ[:, start_index:start_index + sample_len]
                    start_index += sample_len - cfg['sample_overlap']
                    label = np.array([label_index], dtype=np.float32)
                    samples.append(sample)
                    labels.append(label)
                shuffled_index = list(range(len(samples)))
                np.random.shuffle(shuffled_index)
                test_index = shuffled_index[:test_sample_num]
                with h5py.File(saved_file_name, 'a') as hf:
                    hf['test_data'].resize(hf['test_data'].shape[0] + len(test_index), axis=0)
                    hf['test_label'].resize(hf['test_label'].shape[0] + len(test_index), axis=0)
                    hf['test_data'][-len(test_index):] = [samples[i] for i in test_index]
                    hf['test_label'][-len(test_index):] = [labels[i] for i in test_index]
                    for i in test_index:
                        if labels[i][0] not in test_label_index_dict.keys():
                            test_label_index_dict[labels[i][0]] = [i]
                        else:
                            test_label_index_dict[labels[i][0]].append(i)
                print('generating {} test samples'.format(len(test_index)))
    with open('{}train_label_index.pkl'.format(cfg['train_data_dir']), 'wb') as f:
        pickle.dump(train_label_index_dict, f)
    with open('{}test_label_index.pkl'.format(cfg['train_data_dir']), 'wb') as f:
        pickle.dump(test_label_index_dict, f)


if __name__ == '__main__':
    h5_file = cfg['h5_file']
    sample_len = cfg['sample_len']
    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('train_data', (0, 2, sample_len), maxshape=(None, 2, sample_len), dtype='f4')
        hf.create_dataset('train_label', (0, 1), maxshape=(None, 1), dtype='f4')
        hf.create_dataset('test_data', (0, 2, sample_len), maxshape=(None, 2, sample_len), dtype='f4')
        hf.create_dataset('test_label', (0, 1), maxshape=(None, 1), dtype='f4')
    print('Generating train/test samples for value net...')
    gen_dataset(
        data_dir=cfg['train_data_dir'],
        saved_file_name=h5_file,
        sample_len=sample_len,
        train_sample_num=cfg['train_sample_num'],
        test_sample_num=cfg['test_sample_num'],
        train_test_from_same=True
    )
    dataset = SEIDataset(data_file=cfg['h5_file'], split='train')
    mean, std = get_mean_std(dataset, ratio=1)
    with open('{}mean_std.pkl'.format(cfg['train_data_dir']), 'wb') as f:
        pickle.dump([mean, std], f)
    print('mean:{}, std:{}'.format(mean, std))
