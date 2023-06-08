import torch
from torch import nn

from models.common import Conv
from myTricks.CBF import CBF
from myTricks.res_unit import res_unit


class C3(nn.Module):
    # res_unit with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBF(c1, c_, 1, 1)
        self.cv2 = CBF(c1, c_, 1, 1)
        self.cv3 = CBF(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(res_unit(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
