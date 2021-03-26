import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 读取获取数据
batch_size = 300

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 初始化模型参数
# 权重向量784*10    偏差参数向量1*10

num_inputs = 784
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)),
                 dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义softmax函数运行规则
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


# 定义模型

def softmax_net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), w) + b)


# 定义损失函数


def cross_entropy(y_predict, y):
    return -torch.log(y_predict.gather(1, y.view(-1, 1)))


# 计算分类准确率

def accuracy(y_predict, y):
    return (y_predict.argmax(dim=1) == y).float().mean().item()


# 评价模型在数据集data_iter上的准确率

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x_mini, y in data_iter:
        acc_sum += (net(x_mini).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        return acc_sum / n


# print(evaluate_accuracy(test_iter, softmax_net))


# 定义小批量随机梯度函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


num_epochs, lr = 5, 0.01


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


# x = torch.rand(num_inputs, num_outputs)

train_softmax(softmax_net, train_iter, test_iter, cross_entropy, num_epochs,
              batch_size, [w, b], lr)


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
predict_lalels = getlabels(softmax_net(x_mini).argmax(dim=1).numpy())
titles = [true + '\n' + predict for true, predict in zip(true_labels, predict_lalels)]

show_mnist(x_mini[11:19], titles[11:19])
