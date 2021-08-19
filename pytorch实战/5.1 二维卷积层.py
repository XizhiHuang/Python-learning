import torch
from torch import nn


def corr2d(x, k):
    # x为输入数组，k为卷积核
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return y


"""

x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
k = torch.tensor([[0, 1], [2, 3]])
print(corr2d(x, k))

"""


# 定义一个二维卷积层
class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 图像物体边缘检测
# 黑色为0 白色为1
x = torch.ones(6, 8)
x[:, 2:6] = 0
print(x)
k = torch.tensor([[1, -1]])
y = corr2d(x, k)
print(corr2d(x, k))

# 通过数据学习核数组
# 使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K

conv2d = Conv2d(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    y_pred = conv2d(x)
    loss = ((y - y_pred) ** 2).sum()
    loss.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)

    if (i + 1) % 5 == 0:
        print('step %d,loss %f' % (i + 1, loss.item()))

print('weight:',conv2d.weight.data)
print('bias:',conv2d.bias.data)