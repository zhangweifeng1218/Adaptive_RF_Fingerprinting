import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, dropout=0, bias=True, relu=True, wn=False):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layer = nn.Linear(in_dim, out_dim, bias)
            if wn: layer = weight_norm(layer, dim=None)
            layers.append(layer)
            if relu: layers.append(nn.ReLU())

        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layer = nn.Linear(dims[-2], dims[-1], bias)
        if wn: layer = weight_norm(layer, dim=None)
        layers.append(layer)
        if relu: layers.append(nn.ReLU())


        if not wn:
            for m in layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

        self.main = nn.Sequential(*layers)


    def forward(self, x):
        return self.main(x)