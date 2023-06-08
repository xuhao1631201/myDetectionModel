import torch
import torch.nn as nn
class Upsample(nn.Module):
    #表示上采样后的特征图的宽度和高度将是输入特征图的两倍。
    #mode表示进行上采样的方式nearest interpolation
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
