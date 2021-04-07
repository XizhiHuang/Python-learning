import torch
from torch import nn

"""

# 不含模型参数的自定义层
class centerlayer(nn.Module):
    def __init__(self):
        super(centerlayer, self).__init__()

    def forward(self, x):
        return x - x.mean()


layer = centerlayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

net = nn.Sequential(nn.Linear(8, 128), centerlayer())

y = torch.rand(4, 8)
print(net(y))

"""


# 含模型参数的自定义层
class mydense(nn.Module):
    def __init__(self):
        super(mydense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = mydense()
print(net)

x = torch.rand(2, 4)
print(net(x))
