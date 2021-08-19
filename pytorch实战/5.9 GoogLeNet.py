import time
import torch
from torch import nn, optim
import sys
from torch.utils import data
import torch.nn.functional as F
import torchvision

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(inception, self).__init__()
        # 线路1
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        # 线路2
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        # 线路3
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        # 线路4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x_input):
        p1 = F.relu(self.p1_1(x_input))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x_input))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x_input))))
        p4 = F.relu(self.p4_2(self.p4_1(x_input)))
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32),
    inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),
    inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),
    inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),
    inception(in_channels=512, c1=112, c2=(144, 288), c3=(32, 64), c4=64),
    inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
    inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128),
    nn.AdaptiveAvgPool2d(output_size=(1, 1))
)


# 实现对x现状的转换
class flattenlayer(nn.Module):
    def __init__(self):
        super(flattenlayer, self).__init__()

    def forward(self, x_input):
        return x_input.view(x_input.shape[0], -1)


net = nn.Sequential(b1, b2, b3, b4, b5, flattenlayer(), nn.Linear(1024, 10))
# net.cuda()

X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)


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


batch_size = 32
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)


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


def train_nin(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on:', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for x_mini, y_mini in train_iter:
            x_mini = x_mini.cuda()
            y_mini = y_mini.cuda()
            y_pred = net(x_mini.cuda())
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
train_nin(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
