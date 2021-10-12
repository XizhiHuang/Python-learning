# -*- coding: utf-8 -*-
"""
# @Time    : 2021/10/12 9:59
# @Author  : Xizhi Huang
# @FileName: 1.提取光谱到同一文件夹.py
# @Software: PyCharm
"""



import os
import shutil
# 对重采样后的光谱筛选
# 对特定LAI进行筛选

"""
for root, dirs, files in os.walk(r"H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/数据svc"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        data_name = data_list[44:]
        print(data_name)
        if data_list[-3:] == 'sig':
            # shutil.copy("file.txt", "file_copy.txt")
            data_copy_list = "H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/光谱all/" + data_name
            shutil.copy(data_list, data_copy_list)
            print('copy')

print('finish!!!')
"""

# 对重采样数据
for root, dirs, files in os.walk(r"H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/光谱all"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        data_name = data_list[32:]
        print(data_name)
        if data_list[-5:] == 'p.sig':
            # shutil.copy("file.txt", "file_copy.txt")
            data_copy_list = "H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/光谱resample/" + data_name
            shutil.copy(data_list, data_copy_list)
            print('copy')

print('finish!!!')