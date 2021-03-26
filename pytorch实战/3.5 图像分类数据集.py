import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 转换后的格式是C x H x W，第一维是通道数，因为数据集中是灰度图像，
# 所以通道数为1。后面两维分别是图像的高和宽。
feature, label = mnist_train[1]
print(feature.shape, label)


# 获取数据标签
def getlabels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


from IPython import display


def use_jpg_display():
    display.set_matplotlib_formats('jpg')


def show_mnist(images, labels):
    # use_jpg_display()
    _, figs = plt.subplots(1, len(images), figsize=(10, 10))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

"""

# x_mini图像内容，y标签
x_mini, y = [], []
for i in range(10):
    x_mini.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_mnist(x_mini, getlabels(y))

"""

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print(num_workers)
start=time.time()
for x_mini,y in train_iter:
    continue
print('%f sec'%(time.time()-start))