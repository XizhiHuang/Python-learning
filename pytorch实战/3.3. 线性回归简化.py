import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
#print(features[0])
# print(features)
#print(labels[0])


print(labels)



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


# 利用nn定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # 定义forward 前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)

for param in net.parameters():
    print(param)

# 初始化模型参数
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

"""

# 调整学习率 设置为动态
for param_group in optimizer.param_group:
    param_group['lr']*=0.1  #学习率边为上一步的0.1

"""

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for x_mini, y in data_iter:
        output = net(x_mini)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()  # 更新模型
    print('epoch:%d,loss:%f' % (epoch, l.item()))

dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)
