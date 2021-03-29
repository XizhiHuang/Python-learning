import torch
import torch.nn as nn
import numpy as np
import sys
import torchvision
import matplotlib.pyplot as plt
from IPython import display
import torchvision.transforms as transforms
from torch.utils import data
import torch.utils.data

# 训练样本20 输入特征200
n_trains, n_tests, n_inputs = 20, 100, 200

true_w, true_b = torch.ones(n_inputs, 1) * 0.01, 0.05
features = torch.randn((n_trains + n_tests, n_inputs))
print(features)
print(features.shape)
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features = features[:n_trains, :]
test_features = features[n_trains:, :]
train_labels = labels[:n_trains]
test_labels = labels[n_trains:]


# 初始化模型参数
def init_params():
    w = torch.randn((n_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义l2范数正则化
def l2_penalty(w):
    return (w ** 2).sum() / 2


# 构建线性回归函数
def linreg(x, w, b):
    return torch.mm(x, w) + b


# 构建损失函数
def squared_loss(y_predict, y):
    return (y_predict - y.view(y_predict.size())) ** 2 / 2


# 定义小批量随机梯度函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 定义作图函数

def use_jpg_display():
    display.set_matplotlib_formats('jpg')


def set_figsize(figsize=(5.5, 5.5)):
    use_jpg_display()
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_labels, y_labels, x2_vals=None, y2_vals=None,
             legend=None, figsize=(9, 9)):
    set_figsize(figsize)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.semilogy(x_vals, y_vals)  # 使用semilogy表示y轴使用对数尺度
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')  # 使用点来表示
        plt.legend(legend)


# 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net = linreg
loss = squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()
    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        for x_mini, y in train_iter:
            l = loss(net(x_mini, w, b), y) + l2_penalty(w) * lambd
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            sgd([w, b], lr, batch_size)

        train_loss.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_loss.append(loss(net(test_features, w, b), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss',
             range(1, num_epochs + 1), test_loss, ['train', 'test'])
    print('w:', w.norm().item())


# fit_and_plot(lambd=30)


# 使用简化的方式
# 使用weight_decay参数来指定权重衰减超参数

def fit_and_polt_easy(wd):
    net = nn.Linear(n_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)

    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        for x_mini, y in train_iter:
            l = loss(net(x_mini), y).mean()

            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            optimizer_w.step()
            optimizer_b.step()

        train_loss.append(loss(net(train_features), train_labels).mean().item())
        test_loss.append(loss(net(test_features), test_labels).mean().item())

    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss',
             range(1, num_epochs + 1), test_loss, ['train', 'test'], (10, 10))


fit_and_polt_easy(0)
