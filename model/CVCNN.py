import torch
import torch.nn as nn
import torch.nn.functional as F
from model.complexLayers import ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv1d, ComplexConv2d, ComplexLinear
from model.complexFunctions import complex_relu, complex_max_pool2d, complex_avg_pool1d, complex_max_pool1d, complex_abs


class CVCNN(nn.Module):
    def __init__(self, cfg=None):
        super(CVCNN, self).__init__()
        self.conv1 = ComplexConv1d(in_channels=2, out_channels=100, kernel_size=40, stride=20)
        self.conv2 = ComplexConv1d(in_channels=100, out_channels=100, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(100 * 20, 100)
        self.fc2 = nn.Linear(100, cfg['n_classes'])

    def forward(self, x):
        x = x.type(torch.complex64)
        x = self.conv1(x)
        x = complex_relu(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = x.abs()
        x = x.view(-1, 100 * 20)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    from config import cfg
    x = torch.randn(size=(1, 2, 512))
    net = CVCNN(cfg)
    y = net(x)
    print(y.shape)
    from utils import count_parameters
    print(count_parameters(net))