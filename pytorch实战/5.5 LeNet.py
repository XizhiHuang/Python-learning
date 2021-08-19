import time
import torch
from torch import nn, optim
import sys
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=70),
            nn.ReLU(),
            nn.Linear(in_features=70, out_features=10)
        )

    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature.view(x.shape[0], -1))
        return output


net = LeNet()
print(net)

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


def train_LeNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
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
train_LeNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
