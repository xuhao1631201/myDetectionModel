import torch

x = torch.arange(10).reshape(5, 2)
print(x)
y = torch.split(x, [1, 4])
print(y)