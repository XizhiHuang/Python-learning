import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from torch.utils import data
import torch.utils.data
import matplotlib.pyplot as plt
from IPython import display

sys.path.append('D:/python data')

torch.set_default_tensor_type(torch.FloatTensor)

# 利用pandas读取csv数据

train_data = pd.read_csv('D:/python data/kaggle_house/train.csv')
test_data = pd.read_csv('D:/python data/kaggle_house/test.csv')

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4])

# 把训练数据和测试数据一起做联立

train_data_concat = train_data.iloc[:, 1:-1]
test_data_concat = test_data.iloc[:, 1:]
data_concat = [train_data_concat, test_data_concat]
all_features = pd.concat(data_concat)
print(all_features.shape)

# 数据标准化处理
# 将该特征的每个值先减去均值μ再除以标准差σ得到标准化后的每个特征值。
# 对于缺失的特征值，我们将其替换成该特征的均值。
# 这一步是提取那些连续数据numerical data 并对这些数据进行标准化处理
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 离散数据categorical data是可数的，可以计算统计其包含的每个value的个数
# 将离散数值转成指示特征  将原先每个特征维度的各个值转换到新建的列方向上
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
print(all_features.iloc[:, 0:4])

# all_features.to_csv('D:/python data/kaggle_house/1.csv')


# 将pandas的数据转换为numpy形式，进而转换到tensor
print(train_data.shape[0])
print(train_data.shape[1])

n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)

train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 训练模型
loss = torch.nn.MSELoss()


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
        return rmse.item()


# 使用Adam进行优化

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()

    for epoch in range(num_epochs):
        for x_mini, y_mini in train_iter:
            l = loss(net(x_mini.float()), y_mini.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 训练集、验证集、测试集
# 先在训练集上训练得到参数，再到验证集上评价其泛化能力，最后在测试集上评价
# 测试集只使用一次


# K折交叉验证
def get_k_fold_data(k, i, x, y):
    # x是feature y是对应的label
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part = x[idx, :]
        y_part = y[idx]
        if j == i:
            x_valid = x_part
            y_valid = y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return x_train, y_train, x_valid, y_valid


# 定义作图函数

def use_jpg_display():
    display.set_matplotlib_formats('jpg')


def set_figsize(figsize=(3.5, 5, 5)):
    use_jpg_display()
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_labels, y_labels, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 5.5)):
    set_figsize(figsize)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.semilogy(x_vals, y_vals)  # 使用semilogy表示y轴使用对数尺度
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')  # 使用点来表示
        plt.legend(legend)


# 建议k折交叉验证的训练和验证的平均误差

def k_flod(k, x_train, y_train, num_epochs, learning_rate,
           weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net(x_train.shape[1])
        train_loss, valid_loss = train(net, *data,
                                       num_epochs, learning_rate, weight_decay,
                                       batch_size)
        train_l_sum += train_loss[-1]
        valid_l_sum += valid_loss[-1]

        if i == 0:
            semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss_rmse',
                     range(1, num_epochs + 1), valid_loss, ['train', 'valid'], (5, 5))

        print('flod %d,train_rmse %f,valid_rmse %f' % (i, train_loss[-1], valid_loss[-1]))

    return train_l_sum / k, valid_l_sum / k


k, num_epochs, learning_rate, weight_decay, batch_size = 5, 100, 5, 0.01, 64
train_loss, valid_loss = k_flod(k, train_features, train_labels, num_epochs, learning_rate,
                                weight_decay, batch_size)

print('%d-fold valid:avg train_loss_rmse %f,avg valid_loss_rmse %f'
      % (k, train_loss, valid_loss))

# 使用测试数据集进行结果预测 这个数据只能使用一次
def train_and_pred(train_features,train_labels,test_features,test_labels,
                   num_epochs,learning_rate,weight_decay,batch_size):
    net=get_net(train_features.shape[1])
    train_loss,test_loss=train(net,train_features,train_labels,test_features,
                               test_labels,num_epochs,learning_rate,
                               weight_decay,batch_size)
    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss_rmse',
             range(1, num_epochs + 1), valid_loss, ['train', 'test'], (9, 9))
    print('test_loss_rmse:%f'%test_loss[-1])
