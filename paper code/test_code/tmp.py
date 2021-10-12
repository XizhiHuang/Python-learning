



import cv2 as cv



from osgeo import gdal
import numpy as np
from gdalconst import *
import time

"""
1213 0409 0519
ndvi savi rvi evi dvi
"""
input_data = gdal.Open("D:/result/minipic1/1051.tif")
output_path = 'D:/result/resize/1.tif'
path="D:/result/minipic1/1051.tif"
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



#img=cv.imread(path,2)

img1=gdal.Open(path)
img_pil=img1.ReadAsArray(0, 0, 480, 480)
band1 = cv.resize(img_pil[0],(240, 240), interpolation=cv.INTER_NEAREST)
band2 = cv.resize(img_pil[1],(240, 240), interpolation=cv.INTER_NEAREST)
band3 = cv.resize(img_pil[2],(240, 240), interpolation=cv.INTER_NEAREST)
band4 = cv.resize(img_pil[3],(240, 240), interpolation=cv.INTER_NEAREST)
band=[band1,band2,band3,band4]
band=np.array(band)









