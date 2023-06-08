import torch
from torch import nn
class Pooling(nn.Module):
    def __init__(self, inp, oup, reduction=32, H=None,W=None):
        super(Pooling, self).__init__()
        #horizontal pooling
        self.pool_h = nn.AdaptiveAvgPool2d((H, 1))
        #vertical pooling
        self.pool_w = nn.AdaptiveAvgPool2d((1, W))
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W,
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # x_h and  x_w concatenated
        y = torch.cat([x_h, x_w], dim=2)
        #Returns the result
        return y