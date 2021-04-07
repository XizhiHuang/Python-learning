import torch
from torch import nn
from torch.nn import init
"""
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

print(net)
x = torch.rand(3, 4)
print(net(x).sum())

# 访问模型参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

"""

"""
torch.nn.parameter.Parameter，其实这是Tensor的子类，和Tensor不同的是
如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = mymodel()
for name, param in n.named_parameters():
    print(name)
"""



"""

# 对权值进行初始化
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

# 对偏置进行初始化
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


# 自定义初始化
# 令权重有一半概率初始化为0，有另一半概率初始化为[−10,−5]和
# [5,10]两个区间里均匀分布的随机数。

def init_weight_(tensor):
    with torch.no_grad():
        # 从（-10，10）中随机赋值
        tensor.uniform_(-10, 10)
        # (tensor.abs()>=5).float() 如果绝对值大于等于5返回1，否则返回0
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)


"""



# 共享模型参数

linear=nn.Linear(1,1,bias=False)
net=nn.Sequential(linear,linear)
print(net)
for name,param in net.named_parameters():
    init.constant_(param,val=3)
    print(name,param.data)

# 在进行模型共享的时候，参数可以做到共享，但是在计算梯度时，梯度会累加

x=torch.ones(1,1)
y=net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)