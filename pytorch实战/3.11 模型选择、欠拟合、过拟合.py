# 多项式函数拟合实验

import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython import display

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, -5.6], 5
features = torch.rand((n_train + n_test), 1)
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] +
          true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

print(features[:2], poly_features[:2], labels[:2])


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




# 损失函数使用MSE
num_epochs = 100
loss = torch.nn.MSELoss()

# 多项式拟合函数
def fit_and_plot(train_features,test_features,train_labels,test_labels):
    net=torch.nn.Linear(train_features.shape[-1],1)

    batch_size=min(10,train_labels[0])
    dataset=torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

    optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
    train_loss,test_loss=[],[]
    for _ in range(num_epochs):
        for x_mini,y in train_iter:
            l=loss()
