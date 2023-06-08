import torch
import torch.nn as nn
class BN(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.bn1 = nn.BatchNorm2d()
    def forward(self, x):
        x = self.bn1(x)
        return x
