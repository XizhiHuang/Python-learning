import torch
from time import time

"""

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
# 作为标量对两个向量进行计算
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[1] + b[i]
end = time()
print(end - start)

start_d = time()
# 直接对两个向量进行计算
d = a + b
end_d = time()
print(start_d - end_d)

"""

"""

# 简单张量计算
a = torch.ones(3)
b = 10
print(a + b)

"""

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
print(features[0])
# print(features)
print(labels[0])


# print(labels)


def use_jpg_display():
    display.set_matplotlib_formats('jpg')


def set_figsize(figsize=(5, 5)):
    use_jpg_display()
    plt.rcParams['figure.figsize'] = figsize


# set_figsize()
plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.scatter(features[:, 1].numpy(), labels.numpy(), 20)
plt.show()


# 读取数据  每次返回batch_size（批量大小）个随机样本的特征和标签。
def read_data(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取整个样本
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 考虑到最后一次取值可能会超出num_examples的范围
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        # a = features.index_select(0, j)
        # b = labels.index_select(0, j)
        yield features.index_select(0, j), labels.index_select(0, j)
        # return a, b


batch_size = 10
# 配合使用yield 可以使得x和y分别得到赋值       也可使用return，但是使用return就只能对x赋值
for x_mini, y in read_data(batch_size, features, labels):
    print(x_mini)
    print(y)

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
b = torch.zeros(1, dtype=torch.float)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def linreg(x, w, b):
    return torch.mm(x, w) + b


def squared_loss(y_predict, y):
    return (y_predict - y.view(y_predict.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.01
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x_mini, y in read_data(batch_size, features, labels):
        # l为小批量样本的损失
        l = loss(net(x_mini, w, b), y).sum()
        # 小批量的损失对模型参数求梯度
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    # print(train_l.mean())
    print('epoch:%s,loss:%f' % (epoch + 1, train_l.mean().item()))

    print(true_w, '\n', w)
    print(true_b, '\n', b)
