import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/Xizhi Huang/Desktop/lover.JPG')
cv.imshow('img', img)
# 逆时针旋转90度
img_a = np.transpose(img, (2, 0, 1))
# 顺时针旋转90度
# img = cv.flip(img, 1)
# 顺时针旋转270度
# img = cv.flip(img, 1)
#cv.namedWindow('img', cv.WINDOW_AUTOSIZE)
cv.imshow('img3', img_a)
#cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
