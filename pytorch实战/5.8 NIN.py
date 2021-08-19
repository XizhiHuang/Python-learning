import torch
from torch import nn, optim
import sys
from torch.utils import data
import time
import torchvision
import torch.nn.functional as F

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 定义nin_block
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    # 第一个卷积层可以使用自己定义的超参数
    nin_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        nn.ReLU()
    )
    return nin_block


# 定义nin模型

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


net=nn.Sequential(
    nin_block(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(0.5),
    nin_block(in_channels=384,out_channels=10,kernel_size=3,stride=1,padding=1),
    #GlobelAvgPool(),
    nn.AdaptiveAvgPool2d(output_size=(1,1)),
    flattenlayer()
)

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


def train_nin(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
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


lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_nin(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
