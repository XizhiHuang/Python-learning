import torch
from torch import nn, optim
import time
import torch.nn.functional as F
import sys
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def batch_norm(is_training, x_input, gamma, beta, moving_mean, moving_val, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果不是在训练模式，是在预测模式。直接使用传入的移动平均所得的均值和方差
        x_input_hat = (x_input - moving_mean) / torch.sqrt(moving_val + eps)
    else:
        assert len(x_input.shape) in (2, 4)
        if len(x_input.shape) == 2:
            # 此时为全连接层的情况。计算特征维上的均值和方差
            mean = x_input.mean(dim=0)
            val = ((x_input - mean) ** 2).mean(dim=0)
        else:
            # 此时为二维卷积层的情况，计算通道维上（axis=1）的均值和方差
            # 需要保持x_input的形状方便后面做广播运算
            mean = x_input.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            val = ((x_input - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 在训练模式下使用当前的均值和方差做标准化
        x_input_hat = (x_input - mean) / torch.sqrt(val + eps)
        # 使用移动平均更新mean val
        moving_mean = moving_mean * momentum + mean * (1 - momentum)
        moving_val = moving_val * momentum + val * (1 - momentum)

    # 进行拉伸和偏移
    y = gamma * x_input_hat + beta
    return y, moving_mean, moving_val


# 定义一个新的batchnorm层，以此来保存参数和便于调用参数
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        # 对于卷积层num_feature为4，对于全连接层为2
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的参数进行数据初始化
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和参数迭代的参数初始化
        self.moving_mean = torch.zeros(shape)
        self.moving_val = torch.zeros(shape)

    def forward(self, x_input):
        # 将数据转到在GPU上运行
        if self.moving_mean.device != x_input.device:
            self.moving_mean = self.moving_mean.to(x_input.device)
            self.moving_val = self.moving_val.to(x_input.device)
        # 保存更新后的参数
        Y, self.moving_mean, self.moving_val = batch_norm(self.training, x_input, self.gamma, self.beta,
                                                          self.moving_mean, self.moving_val, eps=1e-5, momentum=0.9)
        return Y


# 实现对x现状的转换
class flattenlayer(nn.Module):
    def __init__(self):
        super(flattenlayer, self).__init__()

    def forward(self, x_input):
        return x_input.view(x_input.shape[0], -1)


batch_norm_net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
    BatchNorm(num_features=6, num_dims=4),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    BatchNorm(num_features=16, num_dims=4),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    flattenlayer(),
    nn.Linear(in_features=16 * 4 * 4, out_features=120),
    BatchNorm(num_features=120, num_dims=2),
    nn.ReLU(),
    nn.Linear(in_features=120, out_features=84),
    BatchNorm(num_features=84, num_dims=2),
    nn.ReLU(),
    nn.Linear(in_features=84, out_features=10)
)


"""
X = torch.rand(1, 1, 224, 224)
for name, blk in batch_norm_net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)
"""

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
batch_size = 256

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # 如果没有选定device，就按照net的device来处理
        device = list(net.parameters())[0].device
    acc_sum, n = 0, 0
    with torch.no_grad():
        for x_mini, y_mini in data_iter:
            if isinstance(net, nn.Module):
                net.eval()
                acc_sum += (net(x_mini.to(device)).argmax(dim=1) == y_mini).float().sum().cpu().item()
                net.train()
            n += y_mini.shape[0]
    return acc_sum / n


def train_batchnorm_LeNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on:', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for x_mini, y_mini in train_iter:
            x_mini = x_mini.to(device)
            y_mini = y_mini.to(device)
            y_pred = net(x_mini)
            l = loss(y_pred, y_mini)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_pred.argmax(dim=1) == y_mini).float().sum().item()
            n += y_mini.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch:%d,loss:%f,train_acc:%f,test_acc:%f,time:%fsec' %
              (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(batch_norm_net.parameters(), lr=lr)
train_batchnorm_LeNet(batch_norm_net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
