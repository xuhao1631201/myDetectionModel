import torch
from torch import nn
# BiFPN
#  two feature map ：add operation
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        #Set the learnable parameter nn. The purpose of Parameter is to convert an untrainable type Tensor into a trainable type parameter
        #And the parameter is registered with the host model as part of it,so that it can be automatically optimized together when the parameter is optimized
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.sigmod(weight[0] * x[0] + weight[1] * x[1]))


# three feature map add operation
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # Fast normalized fusion
        return self.conv(self.sigmod(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

