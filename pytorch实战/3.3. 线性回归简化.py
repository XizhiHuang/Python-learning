import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data

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

batch_size = 10

# 读取数据  每次返回batch_size（批量大小）个随机样本的特征和标签。
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量数据 shuffle实现
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for x_mini, y in data_iter:
    print(x_mini)
    print(y)
    break

#class LinearNet(nn.Module):
    #def __init__(self,n_feature):











"""
# 配合使用yield 可以使得x和y分别得到赋值       也可使用return，但是使用return就只能对x赋值
for x_mini, y in read_data(batch_size, features, labels):
    print(x_mini)
    print(y)

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
b = torch.zeros(1, dtype=torch.float)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


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
"""
