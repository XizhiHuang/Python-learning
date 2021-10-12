import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
import os
from os import listdir

root_dir = "D:/rsdata/unet_test/data/"
train_file = os.path.join(root_dir, "train.csv").replace('\\', '/')
val_file = os.path.join(root_dir, "val.csv").replace('\\', '/')
label_colors_file = os.path.join(root_dir, "label_colors.txt").replace('\\', '/')
label_colors_img = os.path.join(root_dir, "LabeledApproved_full/").replace('\\', '/')
means = np.array([103.939, 116.779, 123.68]) / 255.
h, w = 720, 960
train_h = int(h * 2 / 3)  # 480
train_w = int(w * 2 / 3)  # 640
val_h = int(h / 32) * 32  # 704
val_w = w  # 960
num_class = 32
data_length = 70
total_pixel = h * w


class ReadDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5, scale=1.0):
        self.data = pd.read_csv(csv_file)
        self.means = means
        self.n_class = n_class
        self.scale = scale

        self.flip_rate = flip_rate
        self.crop = crop
        if phase == 'train':
            self.new_h = int(train_h * scale)
            self.new_w = int(train_w * scale)
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False
            self.new_h = val_h
            self.new_w = val_w

    def __len__(self):
        return len(self.data)

    # original code use self.data.ix[idx,0], 'ix' is derprecated, use 'loc'. Check https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html about how to use 'loc'
    def __getitem__(self, idx):
        img_name_ori = self.data.loc[idx, 'img']
        img_name = os.path.join(root_dir, img_name_ori).replace('\\', '/')
        img_pil = Image.open(img_name).convert('RGB')

        w_ori, h_ori = img_pil.size
        s_w, s_h = int(self.scale * w_ori), int(self.scale * h_ori)
        assert s_w > 0 and s_h > 0, 'Scale needs to be positive'
        img_pil = img_pil.resize((s_w, s_h), resample=0)
        img = np.array(img_pil)

        label_name = self.data.loc[idx, 'label']
        label_name = os.path.join(root_dir, label_name).replace('\\', '/')
        label = np.load(label_name)
        label_pil = Image.fromarray(label)
        # use `resample=0` to avoid number larger than 31 (number of object classes)
        label_pil = label_pil.resize((s_w, s_h), resample=0)
        label = np.array(label_pil)

        if self.crop:
            h, w, _ = img.shape
            top = random.randint(0, h - self.new_h)
            left = random.randint(0, w - self.new_w)
            img = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        # The fliplr() function is used to flip array in the left/right direction.
        if random.random() < self.flip_rate:
            img = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        img = img / 255.0
        label = torch.from_numpy(label.copy()).long()

        sample = {'img': img, 'label': label, 'img_name': img_name_ori}

        return sample


# every network class needs to inheritate nn.Module and has forward function
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # about BatchNoram2d, https://pytorch.org/docs/stable/nn.html
        # seems original paper didn't mention that
        # About `inplace == true ` https://www.jianshu.com/p/8385aa74e2de
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# `//` means devision and then floor
# Transpose is learning parameter while Up-sampling is no-learning parameters. Using Up-samling for faster inference or training because it does not require to update weight or compute gradient
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    # x_f is the feedfoward network input, x_s is skipped connection layer
    # 1: upsample 2: padding so that two layer share the same size 3:connect two layer
    def forward(self, x_f, x_s):
        # first, upsample them
        x_f = self.up(x_f)

        # size()[0] is batch size, size()[1] is channel, size()[2] is height, [3] is width
        # print(x_f.type(), x_s.type())
        diff_y = x_s.size()[2] - x_f.size()[2]
        diff_x = x_s.size()[3] - x_f.size()[3]

        # make x_f and x_s the same size
        x_f = F.pad(x_f, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        # connect two layers
        # print('shape of x_s and x_f ', x_s.shape, x_f.shape)
        x = torch.cat([x_s, x_f], dim=1)
        # print('in Up, the concatednated x size is ', x.shape)
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# output is 1x1 conv according to the paper
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.out = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.out(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inp = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64)
        self.outc = Out(64, n_classes)

    def forward(self, x):
        x0 = self.inp(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.outc(x)


def parse_label():
    # change label to class index
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    all_color = []
    for idx, line in enumerate(f):
        label = line.split()[-1]
        color = tuple([int(x) for x in line.split()[:-1]])
        all_color.append(color)
        print(label, color, idx)

    return all_color


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('use device ', device)
    net = UNet(3, num_class)
    net = net.float()
    net = net.to(device=device)

    batch_size = 1

    val_data = ReadDataset(csv_file=val_file, phase='val')

    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    Model_Path = 'D:/rsdata/unet_test/data/unet_model.pth'
    net.load_state_dict(torch.load(Model_Path))

    correct = 0
    total = 0
    all_color = parse_label()
    img_show_time = 10
    print('prediction_groundtruth comparison image will refresh after', img_show_time, 'seconds')

    with torch.no_grad():
        for data in val_data_loader:
            img = data['img']
            mask = data['label']
            img_name = data['img_name']

            img = img.to(device=device, dtype=torch.float32)

            # print(img_name)

            outputs = net(img.float())

            outputs = outputs.data.cpu().numpy()

            pred = outputs.transpose(0, 2, 3, 1).reshape(-1, num_class).argmax(axis=1).reshape(batch_size, h, w)

            # print(pred[0].dtype)
            # print(mask[0].dtype)
            # f = plt.figure()
            # f.add_subplot(1,2,1)
            # plt.imshow(mask[0])
            # f.add_subplot(1,2,2)
            # plt.imshow(pred[0])

            acc_counter = 0
            for batch_num in range(batch_size):
                for i in range(h):
                    for j in range(w):
                        if mask[batch_num][i][j] == pred[batch_num][i][j]:
                            acc_counter = acc_counter + 1
            print('current image batch accuracy is ', acc_counter / (batch_size * total_pixel) * 100.0, '%')

            img_np = np.zeros((h, w, 3), dtype=np.uint8)

            for batch_num in range(batch_size):
                for i in range(h):
                    for j in range(w):
                        color = all_color[pred[batch_num, i, j]]
                        img_np[i, j, 0] = color[0]
                        img_np[i, j, 1] = color[1]
                        img_np[i, j, 2] = color[2]

            oimg = Image.fromarray(img_np, 'RGB')
            tmp_root=img_name[0].replace('\\', '/')
            oimg_root=tmp_root.replace('701_StillsRaw_full', 'result')
            oimg.save(oimg_root)
            # original image
            gt_img_path = os.path.join(root_dir, img_name[0]).replace('\\', '/')
            gt_img = Image.open(gt_img_path)

            # split_path[0] image folder name, split_path[1] image name
            split_path = os.path.split(img_name[0])
            # only got image name, split it with 'png'
            split_path = split_path[1].split('.')
            # in the labeled folder, image name has two more chars '_L'
            split_path = split_path[0] + '_L.png'
            gt_label_path = os.path.join(label_colors_img, split_path).replace('\\', '/')
            gt_label = Image.open(gt_label_path)

            f = plt.figure()
            f.add_subplot(1, 3, 1)
            plt.imshow(oimg)
            f.add_subplot(1, 3, 2)
            plt.imshow(gt_img)
            f.add_subplot(1, 3, 3)
            plt.imshow(gt_label)
            plt.pause(img_show_time)
            plt.close()







