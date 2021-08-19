import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import torchvision
from torch.utils import data

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    )
    return blk


# 稠密块由多个conv_block组成，每块使用相同的输出通道数
# 在前向计算时，我们将每块的输入和输出在通道维上连结

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        # 计算输出通道数
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


# 使得输出通道得到了大量增长
blk = DenseBlock(2, 3, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

"""
由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型
复杂度。它通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一
步降低模型复杂度。
"""


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk


blk = transition_block(23, 10)
print(blk(Y).shape)

# 构建DenseNet
DenseNet = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
