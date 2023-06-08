import torch
from torch import nn
from models.common import autopad
from myTricks.FReLU import FReLU
class CBF(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #Using the novel activation function FReLU,
        self.act =FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        #Two-dimensional convolution, regularized BN processing, FReLU activation
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))
