import torch
from torch import nn, optim
import time
import sys
import torchvision

from torch.utils import data

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 定义一个VGG块
def vgg_block(num_convs, in_channels, out_channels):
    block = []
    for i in range(num_convs):
        if i == 0:
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*block)


# 实现对x现状的转换
class flattenlayer(nn.Module):
    def __init__(self):
        super(flattenlayer, self).__init__()

    def forward(self, x_input):
        return x_input.view(x_input.shape[0], -1)


# 定义VGG网络
# conv_arch指定了每个VGG块里卷积层个数和输入输出通道数
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i + 1), vgg_block(num_convs, in_channels, out_channels))

    # 全连接层
    net.add_module('fc', nn.Sequential(
        flattenlayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_fearures = 512 * 7 * 7
fc_hidden_units = 4096

net = vgg(conv_arch, fc_fearures, fc_hidden_units)
#net.cuda()

x = torch.rand(1, 1, 224, 224)

# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
for name, block in net.named_children():
    x = block(x)
    print(name, 'output shape:', x.shape)


# 获取数据
ratio=8



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


def evaluate_accuracy(data_iter, net):
    if  isinstance(net, nn.Module):
        device_gpu = list(net.parameters())[0].device
    acc_sum, n = 0, 0
    with torch.no_grad():
        for x_mini, y_mini in data_iter:
            if isinstance(net, nn.Module):
                net.eval()
                acc_sum += (net(x_mini.to(device_gpu)).argmax(dim=1) == y_mini.to(device_gpu)).float().sum().cpu().item()
                net.train()
            n += y_mini.shape[0]
    return acc_sum / n


def train_vgg(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on:', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for x_mini, y_mini in train_iter:
            #x_mini = x_mini.cuda()
            #y_mini = y_mini.cuda()
            #y_pred = net(x_mini.cuda())
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
train_vgg(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
