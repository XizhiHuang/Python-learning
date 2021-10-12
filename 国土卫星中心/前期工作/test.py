from osgeo import gdal
import numpy as np
import os
from gdalconst import *
#import matplotlib.pyplot as pyplot


# os.chdir("H:\\国土卫星中心文档\\RX检测")
# 可以调用os库也可以不用调用 不调用就直接gdal.Opens(绝对路径)
# os.chdir("D:\\RX检测")
# dataset = gdal.Open("Landsat_8_OLI_Rad_flaash.dat")  # 打开文件
# 这里面gdal.Opens(" ")引号中间写上存储的位置
dataset = gdal.Open("D:\\RX检测\\tmp.dat")

# dir(dataset) 查看哪些方法可以用
# 栅格矩阵的列数
im_width = dataset.RasterXSize
# 栅格矩阵的行数
im_height = dataset.RasterYSize
# 波段数
im_bands = dataset.RasterCount
im_geotrans = dataset.GetGeoTransform()
# 仿射矩阵，左上角像素的大地坐标和像素分辨率(左上角x, x分辨率，仿射变换，左上角y坐标，y分辨率，仿射变换)
# 地图投影信息，字符串表示
im_proj = dataset.GetProjection()
im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

# # 打开波段,逐像元读取value值，并放到矩阵中
# srcImage = dataset
# Band = dataset.GetRasterBand(2)
# NoData = Band.GetNoDataValue()
#
# print(NoData)

type(im_data), im_data.shape

del dataset  # 关闭对象，文件dataset

print("----------------------")
print(im_width)
print(im_height)
print(im_bands)
print(im_geotrans)
print(im_proj)
print("----------------------")
print(im_data)

# 显示栅格数据
print(im_data.shape)
#print(im_data[0, 45, 0])

# 读写数据
print(im_data.dtype)

# 单一波段含有的像元数
pixel_sum = im_height * im_width

# 构建一个N*M的矩阵 N为通道数，M为一个波段下整个的像元数
all_pixel_array = []
for i in range(im_bands):
    for j in range(im_width):
        for k in range(im_height):
            # a=im_data[i,j,k]
            # print(im_data[i,k,j])
            all_pixel_array.append(im_data[i, k, j])

all_pixel_array = np.array(all_pixel_array)
all_pixel_array = all_pixel_array.reshape((im_bands, pixel_sum))

# 求取每行的平均值
all_row_average_pixel_array = []
for i in range(im_bands):
    row_pixel_sum = 0
    for j in range(pixel_sum):
        row_pixel_sum = row_pixel_sum + all_pixel_array[i, j]
    row_pixel_average = row_pixel_sum / pixel_sum
    all_row_average_pixel_array.append(row_pixel_average)

all_row_average_pixel_array = np.array(all_row_average_pixel_array)
all_row_average_pixel_array = all_row_average_pixel_array.reshape((im_bands, 1))

print(all_pixel_array[1, 1])
print(all_pixel_array[1][1])

# 计算N*M矩阵逐像元减去各行均值
all_subtract_pixel_array = all_pixel_array
for i in range(im_bands):
    for j in range(pixel_sum):
        all_subtract_pixel_array[i][j] = all_pixel_array[i][j] - all_row_average_pixel_array[i][0]

# 对数据进行还原，得到与原输入数据一致的通道、行、列数，形成一个band*row*col的三维数组
all_pixel_array_reshape = all_subtract_pixel_array
all_pixel_array_reshape = all_pixel_array_reshape.reshape((im_bands, im_width, im_height))
# len(reshape.shape)
all_pixel_array_result = im_data

for i in range(im_bands):
    for j in range(im_height):
        for k in range(im_width):
            all_pixel_array_result[i][j][k] = all_pixel_array_reshape[i][k][j]

#
# # 创建一个row*col临时数组
# temp_row = im_height
# temp_col = im_width
# temp_array = [0] * temp_row
# for i in range(len(temp_array)):
#     temp_array[i] = [0] * temp_col
# temp_array = np.array(temp_array)
# temp_array=temp_array.astype(float)
# #print(temp_array.dtype)
#
# # for i in range(im_height):
# #     for j in range(im_width):
# #         temp_array[i][j]=im_data[0][i][j]
#
#
# for i in range(im_bands):
#     temp_reverse = temp_array
#     for j in range(im_width):
#         for k in range(im_height):
#             y=reshape[i][j][k]
#             temp_reverse[k][j] = reshape[i][j][k]
#     if test==[]:
#         test.append(temp_reverse)
#         test = np.array(test)
#     else:
#         temp_reverse=np.array(temp_reverse)
#         np.concatenate((test,temp_array))
#
# test = np.array(test)


# 设定局部rx算法的内外窗窗口大小
# 外窗口
win_out = 11
# 内窗口
win_in = 3

# ##########################完成对数据预处理部分##########################

# ##########################    RX算法部分    ##########################

all_pixel_array_result_back = all_pixel_array_result
a = all_pixel_array_result_back - all_pixel_array_result
data = all_pixel_array_result_back

a = im_height
b = im_width
c = im_bands

# 创建一个row*col临时数组
temp_row = im_height
temp_col = im_width
temp_array = [0] * temp_row
for i in range(len(temp_array)):
    temp_array[i] = [0] * temp_col
temp_array = np.array(temp_array)
temp_array = temp_array.astype(float)
# print(temp_array.dtype)

# rx_result = temp_array

rx_result = np.random.rand(im_height, im_width)

t_out = int(win_out / 2)
t_in = int(win_in / 2)
m = win_out * win_out

# 对八边进行填充
# 形成一个band*3row*3col的数组
data_input = np.zeros((im_bands, 3 * im_height, 3 * im_width))

# 将原始数据填入中间
for i in range(im_bands):
    for j in range(im_height, 2 * im_height):
        for k in range(im_width, 2 * im_width):
            data_input[i][j][k] = data[i][j - im_height][k - im_width]

# 将填入的数据向左镜像填充
for i in range(im_bands):
    for j in range(im_height, 2 * im_height):
        for k in range(0, im_width):
            data_input[i][j][k] = data[i][j - im_height][im_width - k - 1]

# 将填入的数据向右镜像填充
for i in range(im_bands):
    for j in range(im_height, 2 * im_height):
        for k in range(2 * im_width, 3 * im_width):
            data_input[i][j][k] = data[i][j - im_height][3 * im_width - k - 1]

# 将上面填充的结果整体向上翻转镜像填充
for i in range(im_bands):
    for j in range(0, im_height):
        for k in range(0, 3 * im_width):
            data_input[i][j][k] = data_input[i][2 * im_height - j - 1][k]

# 将上面填充的结果整体向下翻转镜像填充
for i in range(im_bands):
    for j in range(2 * im_height, 3 * im_height):
        for k in range(0, 3 * im_width):
            data_input[i][j][k] = data_input[i][4 * im_height - j - 1][k]

block = data_input[:, 0:3, 0:3]

# 遍历循环计算相关的rx系数
for i in range(im_bands):
    for j in range(im_width, 2 * im_width):
        for k in range(im_height, 2 * im_height):
            block_copy = data_input.copy()
            block = block_copy[:, k - t_out:k + t_out + 1, j - t_out:j + t_out + 1]
            data_input_copy = data_input.copy()
            y = data_input_copy[:, k, j]
            y_out = y.reshape(1, im_bands)
            # 内窗口处band*（3*3）区域内赋为非数值元素NaN
            block[:, t_out - t_in:t_out + t_in + 1, t_out - t_in:t_out + t_in + 1] = np.NaN
            tmp_block = []
            for z in range(win_out):  # width
                for y in range(win_out):  # height
                    for x in range(im_bands):
                        tmp_block.append(block[x][y][z])
            tmp_block = np.array(tmp_block)
            tmp_block = tmp_block.reshape((m, im_bands))
            block_del_nan = tmp_block[~np.isnan(tmp_block).any(axis=1), :]
            # 对删除nan的数组转置
            block_transpose = np.transpose(block_del_nan)
            sigma = np.matmul(block_transpose, block_del_nan)
            sigma_pinv = np.linalg.pinv(sigma)
            # 计算RX算子
            # 计算y_out的转置
            y_out_transpose = np.transpose(y_out)
            tmp_result = np.matmul(y_out, sigma_pinv)
            tmp_data_array = np.matmul(tmp_result, y_out_transpose)
            tmpdata = tmp_data_array[0, 0]
            # print(k - im_height)
            # print(j - im_width)
            rx_result[k - im_height][j - im_width] = tmpdata
            tmpdata1 = tmp_data_array[0, 0]
            block = np.zeros((im_bands, win_out, win_out))

rx_output_data = rx_result

# pyplot.imshow(rx_output_data)

# ########################## 完  成  RX  算  法##########################

# ##########################    输 出 栅 格    ##########################

# 确定文件的输出格式为tif
driver = gdal.GetDriverByName('GTiff')
# 设置输出路径及其输出文件名
# 写数据的具体位置 具体到文件夹下的具体文件 这个文件是新创建的 命名为hyper_tmp_result.tif
output_filename = 'D:\\RX检测\\hyper_tmp_result624.tif'
# 设置创建的栅格数据格式
output_data_gdal_tif = driver.Create(output_filename, im_width, im_height, 1, gdal.GDT_Float32)
# 写入仿射变换参数
output_data_gdal_tif.SetGeoTransform(im_geotrans)
# 写入投影信息
output_data_gdal_tif.SetProjection(im_proj)
# 写入data数据
output_data_gdal_tif.GetRasterBand(1).WriteArray(rx_output_data)
del output_data_gdal_tif
print('----------')



