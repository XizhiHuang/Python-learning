import torch

"""
创建空的tensor
x = torch.empty(5, 3)
print(x)

随机创建tensor
x = torch.rand(5, 3)
print(x)

创建int全0
x = torch.zeros(5, 3, dtype=torch.int)
print(x)


x = torch.tensor([5.5, 3], dtype=int)
print(x)
"""

"""

x = torch.tensor([[5.5, 3], [5.5, 3]])
print(x)
x = x.new_ones(4, 3, dtype=torch.float)     # 在原有tensor的基础上修改tensor结构
print(x)
x = torch.randn_like(x, dtype=torch.float)  # 指定新的数据类型
print(x)
x = torch.eye(3, 2)
print(x)

# 获取tensor形状
print(x.shape)
print(x.size())
"""

"""
x = torch.zeros(5, 3)
y = torch.ones(5, 3)
print(x)
print(y)
print(x+y)

"""

"""
view改变tensor形状
x = torch.rand(5, 3)
y = x.view(15)
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())

# x = x + 1
x += 1
print(x)
print(y)
"""

"""
x = torch.rand(5, 3)
x_copy = x.clone().view(15)
x += 1
print(x)
print(x_copy)
"""

"""
# 广播机制  对于两个不同形状的tensor进行加减操作
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x - y)
"""

"""
x=torch.ones(5)
print(x)
y=x.numpy()
print(x,y)

x+=1
print(x,y)

y+=1
print(x,y)
"""

"""

import numpy as np

x = np.ones(5)
print(x)
y = torch.from_numpy(x)
print(x, y)

x += 1
print(x, y)

y += 1
print(x, y)

# 使用torch.tensor也可以对numpy转为tensor 但是数据不再共享
z = torch.tensor(x)
x += 1
print(x, z)

"""

"""
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)
# out是一个标量没有方向 零维
print(out)

out.backward()
# out关于x的梯度
print(x.grad)

out2 = x.sum()
print(out2)
out2.backward()
print(x.grad)

# 消除之前存在的梯度
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

"""

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
print(y)
z = y.view(2, 2)
print(z)

v = torch.tensor([[1, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(z)
print(x.grad)
