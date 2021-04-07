# 当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为p，
# 那么有p的概率hi会被清零，有1−p的概率hi会除以1−p做拉伸。丢弃概率是丢弃法的超参数。

import torch
import torch.nn as nn
import sys
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils import data
import torch.utils.data
from torch.nn import init


def dropout(x_drop, drop_prob):
    x_drop = x_drop.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if drop_prob == 0:
        return torch.zeros_like(x_drop)

    mask = (torch.rand(x_drop.shape) < keep_prob).float()
    return mask / keep_prob


"""

验证使用不同dropout的结果

x = torch.arange(16).view(2, 8)
print(x)

print(dropout(x, 0))
print(dropout(x, 0.1))
print(dropout(x, 0.3))

"""

# 使用Fashion-MNIST数据集。
# 定义一个包含两个隐藏层的多层感知机，其中两个隐藏层的输出个数都是256。


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

w1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, dtype=torch.float, requires_grad=True)
w2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, dtype=torch.float, requires_grad=True)
w3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

params = [w1, b1, w2, b2, w3, b3]

for param in params:
    param.requires_grad_(requires_grad=True)

"""
定义模型
使用relu激活函数
第一个隐含层的dropout设置为0.2
第二个隐含层的dropout设置为0.5
通常的建议是把靠近输入层的丢弃概率设得小一点
以下这个模型有问题！！！！！！！！！！！！！！

"""
dropout_prob1, dropout_prob2 = 0.2, 0.5


# 通过参数is_training来判断运行模式为训练还是测试，并只需在训练模式下使用丢弃法


def net(x, is_training=True):
    x = x.view(-1, num_inputs)
    h1 = (torch.matmul(x, w1) + b1).relu()
    # 只在训练模型时使用丢弃法
    if is_training:
        h1 = dropout(h1, dropout_prob1)
    h2 = (torch.matmul(h1, w2) + b2).relu()
    if is_training:
        h2 = dropout(h2, dropout_prob2)
    return torch.matmul(h2, w3) + b3


# 先定义一个线性层
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        # self.linear = nn.Linear(num_inputs, num_outputs)

    # 后续利用网络训练学习需要调用到forward函数
    def forward(self, x_batch):
        return x_batch.view(x_batch.shape[0], -1)


drop_net = nn.Sequential(
    LinearNet(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout_prob2),
    nn.Linear(num_hiddens2, num_outputs)
)

for param in drop_net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)


# 修改模型评估精度评价函数

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x_mini, y_mini in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()  # 使用评估模式，禁用dropout
            acc_sum += (net(x_mini).argmax(dim=1) == y_mini).float().sum().item()
            net.train()  # 改回训练模式，继续训练

        else:
            # 如果net函数中有is_training这个参数，将is_training设置成False
            if 'is_training' in net.__code__.co_varnames:
                acc_sum += (net(x_mini, is_training=False).argmax(dim=1) == y_mini).float().sum().item()
            else:
                acc_sum += (net(x_mini).argmax(dim=1) == y_mini).float().sum().item()

        n += y_mini.shape[0]

    return acc_sum / n


# 训练和测试模型

num_epochs = 5
lr = 10.0
batch_size = 256
loss = torch.nn.CrossEntropyLoss()

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

"""

# 定义小批量随机梯度函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= param.grad * lr / batch_size

"""

optimizer = torch.optim.SGD(drop_net.parameters(), lr=0.5)


# 训练模型
def train_dropout_net(net, train_iter, test_iter, loss, num_epochs,
                      batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x_mini, y in train_iter:
            y_predict = net(x_mini)
            l = loss(y_predict, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()

            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # 更新所有参数

            train_l_sum += l.item()
            train_acc_sum += (y_predict.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d,loss %f,train_acc %f,test_acc %f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_dropout_net(drop_net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
