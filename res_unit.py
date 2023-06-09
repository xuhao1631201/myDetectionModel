import torch
from torch import nn

from models.common import Conv
from myTricks.CBF import CBF
class res_unit(nn.Module):
    # Standard res_unit
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # size of kernels =1，stride=1
        self.cv1 = CBF(c1, c_, 1, 1)
        # size of kernels =3，stride=1
        self.cv2 = CBF(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))