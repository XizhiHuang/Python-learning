import numpy as np
import xlrd
import math

# 先声明一个空list
resArray = []
# 读取文件
data = xlrd.open_workbook(filename=r"H:/国土卫星中心文档/包头光谱数据/testdata.xlsx")
# 按索引获取工作表，0就是工作表1
table = data.sheet_by_index(0)
# table.nrows表示总行数
for i in range(table.nrows):
    # 读取每行数据，保存在line里面，line是list
    line = table.row_values(i)
    # 将line加入到resArray中，resArray是二维list
    resArray.append(line)
# 将resArray从二维list变成数组
resArray = np.array(resArray)
print(resArray.shape)
# print(resArray[2,2])

LAI = [3.03
    , 3.287
    , 2.716
    , 3.118
    , 2.623
    , 2.761
    , 2.092
    , 2.605
    , 2.75
    , 3.701
    , 3.543
    , 4.848
    , 4.652
    , 4.458
    , 4.872
    , 4.012
    , 4.315
    , 4.42
    , 3.456
    , 3.316
    , 2.625
    , 2.337
    , 3.61
    , 3.477
    , 3.918
    , 4.796
    , 4.6
    , 4.864
    , 4.896
    , 4.644
    , 4.427
    , 4.827
    , 4.7
    , 3.467
    , 4.569
    , 4.359
    , 4.576
    , 4.664
    , 4.71
    , 3.949
    , 4.35
    , 4.168
    , 3.916]
# print(LAI[1])

average_array = []
for i in range(0, 129, 3):
    # print(resArray[:,i])
    average_temp = (resArray[:, i] + resArray[:, i + 1] + resArray[:, i + 2]) / 3
    average_array.append(average_temp)

average_array = np.array(average_array)
print(average_array.shape)
print(average_array[1, :])

print("**********")
print(average_array[0, 280])
print(average_array[0, 431])

ndvi_predict_array = []

for i in range(43):
    for j in range(280, 431):
        for k in range(431, 1151):
            red_value = average_array[i, j]
            nir_value = average_array[i, k]
            temp_ndvi = (nir_value - red_value) / (nir_value + red_value)
            ndvi_predict_array.append(temp_ndvi)

ndvi_predict_array = np.array(ndvi_predict_array)
re_ndvi_pre = ndvi_predict_array.reshape((43, -1))
print(ndvi_predict_array.shape)
print(re_ndvi_pre.shape)
print(re_ndvi_pre[:, 0])
print(LAI)
# average_ndvi_pre_index

average_ndvi_pre_array = []
# sum_ndvi_pre = 0

for j in range(108720):
    sum_ndvi_pre = 0
    for i in range(43):
        sum_ndvi_pre = sum_ndvi_pre + re_ndvi_pre[i][j]
    average_ndvi_pre = sum_ndvi_pre / 43
    average_ndvi_pre_array.append(average_ndvi_pre)

average_ndvi_pre_array = np.array(average_ndvi_pre_array)
# re_sum_ndvi_pre = sum_ndvi_pre_array.reshape((43, -1))
print(average_ndvi_pre_array[726])

# 真实与预测的平均和 分子
sum_true_pre_array = []

# sum_true_pre = 0

for j in range(108720):
    sum_true_pre = 0
    for i in range(43):
        # true_pre = math.pow((LAI[i] - average_ndvi_pre_array[j]), 2)
        # true_pre = (LAI[i] - average_ndvi_pre_array[j])*(LAI[i] - average_ndvi_pre_array[j])
        # sum_true_pre = sum_true_pre + true_pre

        true_pre = math.pow((LAI[i] - re_ndvi_pre[i][j]), 2)
        sum_true_pre = sum_true_pre + true_pre

    sum_true_pre_array.append(sum_true_pre)
sum_true_pre_array = np.array(sum_true_pre_array)
print('****')
print(sum_true_pre_array[6])
print(sum_true_pre_array[12224])

# 真实与平均的平均和 分母
sum_true_average_array = []

# sum_true_average = 0

for j in range(108720):
    sum_true_average = 0
    for i in range(43):
        # true_average = math.pow((LAI[i] - re_ndvi_pre[i][j]), 2)
        # sum_true_average = sum_true_average + true_average

        true_average = math.pow((LAI[i] - average_ndvi_pre_array[j]), 2)
        sum_true_average = sum_true_average + true_average

    sum_true_average_array.append(sum_true_average)
sum_true_average_array = np.array(sum_true_average_array)
print(sum_true_average_array[12224])

# 计算R2

r2_array = []

for i in range(108720):
    r2 = 1-sum_true_pre_array[i] / sum_true_average_array[i]
    #r2 = float(1 - r2_temp)
    r2_array.append(r2)
r2_array = np.array(r2_array)
print("****************")
print(r2_array)

max = -1
for i in range(108720):
    if r2_array[i] >= max:
        max = r2_array[i]
        max_index = i + 1
    else:
        max = max

print(max)
print(max_index)

# 显示输出具体的波段组合
red_index = max_index // 720
nir_index = max_index % 720

# print('选择的红色波段为：')
