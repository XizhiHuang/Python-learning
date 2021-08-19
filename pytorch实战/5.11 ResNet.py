import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import torchvision
from torch.utils import data

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 定义残差类
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


blk = Residual(3, 8, use_1x1conv=True, stride=2)
X = torch.rand((4, 3, 6, 6))
print(X)
print(X.shape)

print(blk(X).shape)

"""
ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、步幅为2的7×77×7卷积层
后接步幅为2的3×33×3的最大池化层。不同之处在于ResNet每个卷积层后增加的批量归一化层。
"""

ResNet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        # 第一个模块的通道数同输入通道数一致
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels=in_channels, out_channels=out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(in_channels=out_channels, out_channels=out_channels))
    return nn.Sequential(*blk)


# 每个模块使用两个残差块
ResNet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
ResNet.add_module("resnet_block2", resnet_block(64, 128, 2))
ResNet.add_module("resnet_block3", resnet_block(128, 256, 2))
ResNet.add_module("resnet_block4", resnet_block(256, 512, 2))


class GlobelAvgPool(nn.Module):
    def __init__(self):
        super(GlobelAvgPool, self).__init__()

    def forward(self, x_input):
        return F.avg_pool2d(x_input, kernel_size=x_input.size()[2:])


# 实现对x现状的转换
class flattenlayer(nn.Module):
    def __init__(self):
        super(flattenlayer, self).__init__()

    def forward(self, x_input):
        return x_input.view(x_input.shape[0], -1)


# 加入全局平均池化层后接上全连接层输出
ResNet.add_module("global_avg_pool", GlobelAvgPool())
ResNet.add_module("fc", nn.Sequential(flattenlayer(), nn.Linear(512, 10)))

X = torch.rand((1, 1, 224, 224))
for name, layer in ResNet.named_children():
    X = layer(X)
    print(name, 'output shape:\t', X.shape)

batch_size = 256


# 加载数据
def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter


train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=None)

lr = 0.001
num_epochs = 5
optimizer = torch.optim.Adam(ResNet.parameters(), lr=lr)


def evaluate_accuracy(data_iter, net):
    if isinstance(net, nn.Module):
        device_gpu = list(net.parameters())[0].device
    acc_sum, n = 0, 0
    with torch.no_grad():
        for x_mini, y_mini in data_iter:
            if isinstance(net, nn.Module):
                net.eval()
                acc_sum += (net(x_mini.to(device_gpu)).argmax(dim=1) == y_mini.to(
                    device_gpu)).float().sum().cpu().item()
                net.train()
            n += y_mini.shape[0]
    return acc_sum / n


def train_resnet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on:', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for x_mini, y_mini in train_iter:
            # x_mini = x_mini.cuda()
            # y_mini = y_mini.cuda()
            # y_pred = net(x_mini.cuda())
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


train_resnet(ResNet, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
