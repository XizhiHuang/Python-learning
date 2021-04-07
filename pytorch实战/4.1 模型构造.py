import torch
from torch import nn
from torch.nn import functional


class mlp(nn.Module):

    # 这个只是定义相关的操作 具体执行步骤按forward函数来
    def __init__(self):
        super(mlp, self).__init__()
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


"""

y = torch.rand(2, 784)
net = mlp()
print(net)
net(y)
print(net(y))

"""


# sequential类
# 它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数
# 来逐一添加Module的实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。


# 构造复杂的模型
# 通过get_constant函数创建训练中不被迭代的参数，即常数参数。在前向计算中，
# 除了使用创建的常数参数外，我们还使用Tensor的函数和Python的控制流，并多次调用相同的层

class fancy_mlp(nn.Module):
    def __init__(self):
        super(fancy_mlp, self).__init__()

        # 定义一个常数参量 不存在梯度 不可被训练
        self.rand_weight = torch.rand((20, 20), dtype=torch.float, requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层，与第一步使用的全连接层参数保持一致
        x = self.linear(x)

        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x = x / 2
        if x.norm().item() < 0.8:
            x = x * 10

        return x.sum()


x = torch.rand(2, 20)
net = fancy_mlp()
print(net)
net(x)
print(net(x))
