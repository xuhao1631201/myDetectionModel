
import torch
from torch import nn
class F1_Conv2d(nn.Module):
    def __init__(self, inp, oup, reduction=32, H=None, W=None):
        super(F1_Conv2d, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((H, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, W))
        mip = max(8, inp // reduction)
        # F1 convolution:kernel_size=1, stride=1, padding=0
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        #input feature map
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        return y
