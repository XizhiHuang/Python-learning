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

# 遍历文件夹对每个数据进行读取，并将结果保存到excel表格中


for root, dirs, files in os.walk(r"H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/光谱sig"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        if data_list[-3:] == 'sig':
            with open(data_list, "r") as f:
                # data=f.read()
                lines = f.readlines()
                # for line in lines:
                # print(line)

                # 对读取的数据进行分割

                # 31行开始获取光谱数据

                # 创建一个excel文件 暂命名为book
                book = xlwt.Workbook(encoding='utf-8', style_compression=0)
                # 创建一个sheet
                sheet = book.add_sheet('data', cell_overwrite_ok=True)
                # 向表中添加数据标题
                # 其中的'0-行, 0-列'指定表中的单元，'X'是向该单元写入的内容
                sheet.write(0, 0, 'band')
                sheet.write(0, 1, 'ref')

                i = 1
                n = 1
                for line in lines:
                    i = i + 1
                    if 2197 > i > 95:
                        tmp_line = line.split('  ')
                        # 写入data
                        sheet.write(n, 0, tmp_line[0])
                        a=tmp_line[3]
                        b=a[:-1]
                        sheet.write(n, 1, b)
                        n = n + 1

                # 保存excel
                save_list = data_list + '.xls'
                book.save(save_list)

print('finish!!!')
