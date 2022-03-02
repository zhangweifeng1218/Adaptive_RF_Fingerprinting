
from torch import nn


def Conv1d(in_channels, kernel_num, kernel_size=3, stride=2):
    # [N, in_channels, L]
    block = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=kernel_num, kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2), stride=stride),
        nn.BatchNorm1d(kernel_num),
        nn.ReLU()
    )
    return block  # [N, kernel_num, Lout]
