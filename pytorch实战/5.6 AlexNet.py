import torch
import time
from torch import nn, optim
import sys
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # padding是同时对上下左右两个方向进行的扩展
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature.view(x.shape[0], -1))
        return output


net = AlexNet()

print(net)


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


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)


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


def train_AlexNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
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
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_AlexNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
