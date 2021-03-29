import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义模型

num_inputs, num_outputs, num_hiddens_1, num_hiddens_2 = 784, 10, 256, 128


# 先定义一个线性层
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        # self.linear = nn.Linear(num_inputs, num_outputs)

    # 后续利用网络训练学习需要调用到forward函数
    def forward(self, x_batch):
        return x_batch.view(x_batch.shape[0], -1)

# 通过修改里面的网络模型结构即可实现对对不同网络模型的构建
mlp_net = nn.Sequential(
    LinearNet(),
    nn.Linear(num_inputs, num_hiddens_1),
    nn.ReLU(),
    nn.Linear(num_hiddens_1, num_hiddens_2),
    nn.ReLU(),
    nn.Linear(num_hiddens_2, num_outputs)
)

for params in mlp_net.parameters():
    init.normal_(params, mean=0, std=0.01)

# 读取数据训练模型


mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 读取获取数据
batch_size = 256

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(mlp_net.parameters(), lr=0.6)

num_epochs = 5


# 评价模型在数据集data_iter上的准确率

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x_mini, y in data_iter:
        acc_sum += (net(x_mini).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        return acc_sum / n


def train_mlp(net, train_iter, test_iter, loss, num_epochs,
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


train_mlp(mlp_net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
