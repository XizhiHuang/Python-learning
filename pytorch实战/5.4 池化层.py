import torch
from torch import nn


def pooling2d(x, pooling_size, mode='max'):
    x = x.float()
    p_h, p_w = pooling_size
    y = torch.zeros(x.shape[0] - p_h + 1, x.shape[1] - p_w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode == 'max':
                y[i, j] = x[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                y[i, j] = x[i:i + p_h, j:j + p_w].mean()
    return y


"""

x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pooling2d(x, (2, 2), 'avg'))

"""

# 填充和步幅
x = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
print(x)

# pool2d=nn.MaxPool2d(3)
# pool2d=nn.MaxPool2d(3,padding=1,stride=2)
# pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))

# print(pool2d(x))

"""

# 使用stack会新增加一个维度
# 使用cat只是在不同维度上进行属性连接
# 是否创建一个新的维度是两者的区别
x1=torch.cat((x,x+1),dim=1)
print(x1.shape)

x2=torch.stack((x,x+1),dim=1)
print(x2.shape)

"""


x=torch.cat((x,x+1),dim=1)
pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(x))
