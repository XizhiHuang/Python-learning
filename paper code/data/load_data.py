"""
class ReadDataset(Dataset):
    def __init__(self, 参数...):

    def __len__(self, 参数...):
        ...
        return 数据长度

    def __getitem__(self, 参数...):
        ...
        return 字典
    __len__需要返回一个表示数据长度的整型量，__getitem__需要返回一个字典。
    ReadDataset这个类名是自定义的，继承了Dataset即可
"""

import torch
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os
import logging

all_image_list=[]

class ReadDataset(Dataset):
    def __init__(self, imgs_dir, data_label):
        self.imgs_dir = imgs_dir
        self.ids = data_label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        imgs_file = self.imgs_dir + str(idx) + '/' + str(i) + '.png'
        # convert('L')将图片转为灰度图
        img = Image.open(imgs_file).convert('L')
        img = np.array(img)
        img = img.reshape(1, 28, 28)
        # 将图片转为0-1之间
        if img.max() > 1:
            img = img / 255
        return {'image': torch.from_numpy(img), 'label': torch.tensor(idx)}

    """
    构造函数__init__里我们有两个参数，一个是imgs_dir，图像地址，另一个是我们之前创建的列表data_label，
    赋值给self.ids. __len__()仅仅是返回了data_label的长度。有趣的是__getitem__函数，我们看到这个函数的参数是i，
    传入了i之后，我们首先根据ids找到它对应的图像里所标识的数字，继而根据找到图像并转化为黑白。之后再转化为np，再reshape。
    原图像读进来本来是28X28，但是根据网络的要求，输入需要是图像通道数X图像尺寸，黑白图片通道为1，
    所以我们reshape为1X28X28。最后图像的像素点的灰度值归一化到0到1.因为我们要使用cross entropy代价函数来训练，
    根据官网，要求cross entropy的矩阵输入的值为0到1。返回的内容格式必须是字典，
    我们这儿字典的内容图像和图像内对应的数字(label)是这个getitem函数如果调用，最终达到的目的就是，
    假如我在代码中输入A = __getitem__(0)，我就应该能得到0.png对应的那张图像，获取图像的方式就是A['image']，
    获取图像是数字几的方式是A['label']
    """


# 定义网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
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


data_length = 60000
data_label = [-1] * data_length
prev_dir = 'D:/rsdata/mnist_png/training/'
after_dir = '.png'

# 对每个图片添加上标签
for id in range(10):
    id_string = str(id)
    tmp=prev_dir+id_string+'/*.png'
    #for filename in glob(prev_dir + id_string + '/*.png'):
    for filename in glob(tmp):
        replace_a=prev_dir + id_string + '\\'
        position = filename.replace(replace_a, '')
        position = position.replace(after_dir, '')
        data_label[int(position)] = id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use device ', device)

net = LeNet()
net=net.float()
net.to(device=device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

dir_imgs = 'D:/rsdata/mnist_png/training/'
all_data = ReadDataset(dir_imgs, data_label)

batch_size = 8

train_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
print("start training")

for epoch in range(1):
    net.train()

    epoch_loss=0.0
    batch_num=0
    for training_batch in train_loader:
        batch_num=batch_num+1

        images=training_batch['image']
        labels=training_batch['label']

        optimizer.zero_grad()

        outputs=net(images.float())

        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        epoch_loss=epoch_loss+loss.item()

        if batch_num %200==199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_num + 1, epoch_loss / 2000))
            epoch_loss=0.0

print('finish training')
path='D:/rsdata/lenet.pth'
torch.save(net.state_dict(),path)


print('start test')

test_data_length=10000
test_data_label=[-1]*test_data_length
test_prev_dir='D:/rsdata/mnist_png/testing/'
test_after_dir = '.png'

# 对每个图片添加上标签
for id in range(10):
    id_string = str(id)
    tmp=test_prev_dir+id_string+'/*.png'
    #for filename in glob(prev_dir + id_string + '/*.png'):
    for filename in glob(tmp):
        replace_a=test_prev_dir + id_string + '\\'
        position = filename.replace(replace_a, '')
        position = position.replace(test_after_dir, '')
        test_data_label[int(position)] = id

test_net=LeNet()
test_net=test_net.float()

test_all_data=ReadDataset(test_prev_dir,test_data_label)

test_batch_size=4
test_loader=DataLoader(test_all_data,batch_size=test_batch_size,shuffle=True)

print("to test")

test_path='D:/rsdata/lenet.pth'
test_net.load_state_dict(torch.load(test_path))

correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        test_images=data['image']
        test_labels=data['label']

        test_output=test_net(test_images.float())

        _,predict=torch.max(test_output,1)

        total=total+test_labels.size(0)

        correct=correct+(predict==test_labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))



