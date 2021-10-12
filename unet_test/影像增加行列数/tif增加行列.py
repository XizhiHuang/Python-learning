"""
gdal创建一个tif文件
读输入的原始tif的坐标空间信息
除有效区域外全部处理为NoData NoData = -3.402823e+38
有效区域全部写入输入原始tif值
"""

from osgeo import gdal
import numpy as np
from gdalconst import *
import time

"""
1213 0409 0519
ndvi savi rvi evi dvi
"""
input_data = gdal.Open("D:/result/tmp1/1.tif")
output_path = 'D:/result/resize/1.tif'

# dir(dataset) 查看哪些方法可以用
# 栅格矩阵的列数
im_width = input_data.RasterXSize
# 栅格矩阵的行数
im_height = input_data.RasterYSize
# 波段数
im_bands = input_data.RasterCount
im_geotrans = input_data.GetGeoTransform()
# 仿射矩阵，左上角像素的大地坐标和像素分辨率(左上角x, x分辨率，仿射变换，左上角y坐标，y分辨率，仿射变换)
# 地图投影信息，字符串表示
im_proj = input_data.GetProjection()
im_data = input_data.ReadAsArray(0, 0, im_width, im_height)
a = im_data.dtype.name

"""
针对单一波段
# 创建一个新的tif文件用来
# 创建的文件格式
format = "GTiff"
driver = gdal.GetDriverByName(format)
# 创建文件的路径
resultPath = output_path
# 创建输出
output_cols = 18240
output_rows = 17280
output_band = 1
output_data = driver.Create(resultPath, output_cols, output_rows, output_band, GDT_Float32)
output_data.SetProjection(im_proj)
# 设置初始像元位置
# 上下共补274个像元，各187个
# 左右共补4个像元，各2个
x_row = im_geotrans[0]
y_row = im_geotrans[3]
# 转换起始坐标
x_output = x_row - 2
y_output = y_row + 187
output_data.SetGeoTransform([x_output, 1.0, 0.0, y_output, 0.0, -1.0])
# 写入数据
raster_data = np.zeros((output_rows, output_cols))

star_time = time.time()

# 将数组全部写为NoData
NoData = -3.402823e+38
for i in range(0, output_rows):
    for j in range(0, output_cols):
        raster_data[i][j] = 0.0

# 将原始数据写入到影像输入栅格中
raster_data[187:187 + im_height, 2:2 + im_width] = im_data

output_data.GetRasterBand(1).WriteArray(raster_data)

output_data = None

end_time = time.time() - star_time
print('finish')
print(end_time)

"""


# 针对多波段
# 创建一个新的tif文件用来
# 创建的文件格式
format = "GTiff"
driver = gdal.GetDriverByName(format)
# 创建文件的路径
resultPath = output_path
# 创建输出
output_cols = 18240
output_rows = 17280
output_band = 3
output_data = driver.Create(resultPath, output_cols, output_rows, output_band, GDT_Int16)
output_data.SetProjection(im_proj)
# 设置初始像元位置
# 上下共补274个像元，各187个
# 左右共补4个像元，各2个
x_row = im_geotrans[0]
y_row = im_geotrans[3]
# 转换起始坐标
x_output = x_row - 2
y_output = y_row + 187
output_data.SetGeoTransform([x_output, 1.0, 0.0, y_output, 0.0, -1.0])
# 写入数据
raster_data_1 = np.zeros((output_rows, output_cols))
raster_data_2 = np.zeros((output_rows, output_cols))
raster_data_3 = np.zeros((output_rows, output_cols))
#raster_data_4 = np.zeros((output_rows, output_cols))

star_time = time.time()

"""
# 将数组全部写为NoData
NoData = -3.402823e+38
for i in range(0, output_band):
    for j in range(0, output_rows):
        for k in range(0, output_cols):
            raster_data[i][j][k] = 0.0
"""


# 将原始数据写入到影像输入栅格中
raster_data_1[187:187 + im_height, 2:2 + im_width] = im_data[0]
raster_data_2[187:187 + im_height, 2:2 + im_width] = im_data[1]
raster_data_3[187:187 + im_height, 2:2 + im_width] = im_data[2]
#raster_data_4[187:187 + im_height, 2:2 + im_width] = im_data[3]


output_data.GetRasterBand(1).WriteArray(raster_data_1)
output_data.GetRasterBand(2).WriteArray(raster_data_2)
output_data.GetRasterBand(3).WriteArray(raster_data_3)
#output_data.GetRasterBand(4).WriteArray(raster_data_4)

output_data = None

end_time = time.time() - star_time
print('finish')
print(end_time)
