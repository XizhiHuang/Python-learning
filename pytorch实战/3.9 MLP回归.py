import torch
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

# 定义模型参数
# 设置超参数隐藏单元个数为256

num_inputs, num_outputs, num_hiddens_1, num_hiddens_2 = 784, 10, 128, 64

w1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens_1)), dtype=torch.float)
b1 = torch.zeros(num_hiddens_1, dtype=torch.float)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens_1, num_hiddens_2)), dtype=torch.float)
b2 = torch.zeros(num_hiddens_2, dtype=torch.float)
w3 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens_2, num_outputs)), dtype=torch.float)
b3 = torch.zeros(num_outputs, dtype=torch.float)

# 对于可以可变参数一定要进行定义梯度！！！！！
params = [w1, b1, w2, b2, w3, b3]
for param in params:
    param.requires_grad_(requires_grad=True)

# 定义激活函数
# 使用max函数来调用ReLU

"""
def ReLU(x):
   return torch.max(input=x,other=torch.tensor(0.0)) 
"""


# 和上面的relu表达的含义一样，这样更便于理解
def ReLU(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


# 定义网络模型
# torch.mm和torch.matmul可以近似等价，前者用在二维矩阵，后者可以多维
def mlp_net(x):
    x = x.view((-1, num_inputs))
    h1 = ReLU(torch.matmul(x, w1) + b1)
    h2 = ReLU(torch.matmul(h1, w2) + b2)
    return ReLU(torch.matmul(h2, w3) + b3)


# 定义损失函数 交叉熵 直接集成在torch.nn里面
loss = torch.nn.CrossEntropyLoss()


# 训练模型

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


num_epochs = 10
lr = 55
train_mlp(mlp_net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


# 获取数据标签
def getlabels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 展示结果
def show_mnist(images, labels):
    # use_jpg_display()
    _, figs = plt.subplots(1, len(images), figsize=(10, 10))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


x_mini, y = iter(test_iter).next()
true_labels = getlabels(y.numpy())
predict_lalels = getlabels(mlp_net(x_mini).argmax(dim=1).numpy())
titles = [true + '\n' + predict for true, predict in zip(true_labels, predict_lalels)]

show_mnist(x_mini[11:19], titles[11:19])
