import torch
from torch import nn
import sys


def corr2d(x, k):
    # x为输入数组，k为卷积核
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return y


def corr2d_multi_in(x, k):
    res = corr2d(x[0, :, :], k[0, :, :])
    for i in range(1, x.shape[0]):
        res += corr2d(x[i, :, :], k[i, :, :])
    return res


x = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
k = torch.tensor([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])
print(corr2d_multi_in(x, k))

# 使用torch.stack使得维度加1
K = torch.stack([k, k + 1, k + 2])
print(K.shape)


def corr2d_multi_out(x, K):
    return torch.stack([corr2d_multi_in(x, k) for k in K])


print(corr2d_multi_out(x, K))


def corr2d_multi_in_out_1x1(x, k):
    c_i, h, w = x.shape
    c_o = k.shape[0]
    x = x.view(c_i, h * w)
    k = k.view(c_o, c_i)
    y = torch.mm(k, x)
    return y.view(c_o, h, w)


a = torch.rand(3, 3, 3)
b = torch.rand(2, 3, 1, 1)

y1 = corr2d_multi_out(a, b)
y2 = corr2d_multi_in_out_1x1(a, b)

print((y1 - y2).norm().item() < 1e-6)
