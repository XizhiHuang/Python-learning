import torch
from torch import nn


def comp_conv2d(conv2d, x):
    x = x.view((1, 1) + x.shape)
    y = conv2d(x)
    return y.view(y.shape[2:])


# conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)

conv2d=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),stride=2)

x = torch.rand(8, 8)
m=comp_conv2d(conv2d, x)
print(m.shape)
