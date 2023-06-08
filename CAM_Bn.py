
import torch
from torch import nn
class CAM_Bn(nn.Module):
    def __init__(self, inp, oup, reduction=32, H=None,W=None):
        super(CAM_Bn, self).__init__()
        #horizontal pooling
        self.pool_h = nn.AdaptiveAvgPool2d((H, 1))
        #vertical pooling
        self.pool_w = nn.AdaptiveAvgPool2d((1, W))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        #Bn operation
        self.bn1 = nn.BatchNorm2d(mip)
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
        y = self.bn1(y)
        return y
