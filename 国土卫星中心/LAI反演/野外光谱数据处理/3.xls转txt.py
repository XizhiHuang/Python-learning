# -*- coding: utf-8 -*-
"""
# @Time    : 2021/10/12 11:15
# @Author  : Xizhi Huang
# @FileName: 3.xls转txt.py
# @Software: PyCharm
"""
# -*- coding: utf-8 -*-
"""
# @Time    : 2021/10/12 10:39
# @Author  : Xizhi Huang
# @FileName: 2.sig转xls.py
# @Software: PyCharm
"""

import os
import xlrd
import xlwt
import pandas as pd
import numpy as np

# 遍历文件夹对每个数据进行读取，并将结果保存到excel表格中

"""
import pandas as pd

df = pd.read_excel('file1.xlsx', sheetname='Sheet1', header=None)		# 使用pandas模块读取数据
print('开始写入txt文件...')
df.to_csv('file2.txt', header=None, sep=',', index=False)		# 写入，逗号分隔
print('文件写入成功!')
"""



for root, dirs, files in os.walk(r"H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/光谱xls"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        df =pd.read_excel(data_list,  header=None)

        data_out=data_list.replace('xls','txt')
        f = open(data_out,'w')

        #df.to_csv(f, header=None, sep=' ',  line_terminator="\n",index=False)
        df.to_csv(f, header=None, sep='\t', index=False)


print('finish!!!')
