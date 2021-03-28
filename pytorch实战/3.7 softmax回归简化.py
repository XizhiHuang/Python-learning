import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

batch_size = 256

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 上一步获取的data是一个（batch_size,1,28,28)的数据，需要将其转换为(batch_size,784)

num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    # 后续利用网络训练学习需要调用到forward函数
    def forward(self, x_batch):
        y = self.linear(x_batch.view(x_batch.shape[0], -1))
        return y


# 下面这个net并没有调用forward函数
net = LinearNet(num_inputs, num_outputs)

print(net)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 赋值为常量

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 定义优化算法 sgd
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练模型
num_epochs = 5


# 评价模型在数据集data_iter上的准确率

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x_mini, y in data_iter:
        acc_sum += (net(x_mini).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        return acc_sum / n


# 定义小批量随机梯度函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_softmax(net, train_iter, test_iter, loss, num_epochs,
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


train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
