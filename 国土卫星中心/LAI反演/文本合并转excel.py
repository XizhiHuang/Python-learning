import os
import xlrd
from xlutils.copy import copy
import xlwt

with open("H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI加密/lo93r0001_0.3_0_0001.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文件
    print(data)

for root, dirs, files in os.walk(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/test"):
    tmp_1 = 0
    for file in files:
        # 获取文件的路径及其文件列表
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        with open(data_list, "r")as f:
            # with open(data_list, encoding='utf-8').read() as f:
            lines = f.readlines()
            # 筛选出lai文件
            # LAI_value = float(data_list[50:53])
            LAI_value = float(data_list[44:47])
            LAI_name = data_list[-25:]
            tmp_1 = tmp_1 + 1
            # 打开创建的excel文件
            read_data = xlrd.open_workbook(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI_data.xls", formatting_info=True)
            print(read_data.sheet_names())
            booksheet = read_data.sheet_by_index(0)
            cell_value = booksheet.cell_value(4, 4)
            print(cell_value)
            tmp_read_data = copy(read_data)
            sheet = tmp_read_data.get_sheet(0)
            sheet.write(0, 0, '波长')

            sheet.write(0, tmp_1, LAI_name)
            i = 1
            n = 1
            for line in lines:
                i = i + 1
                if i > 2:
                    tmp_line = line.split(' ')
                    # 写入data
                    sheet.write(n, 0, tmp_line[0])
                    sheet.write(n, 1, tmp_line[1])
                    n = n + 1

# 保存数据
tmp_read_data.save("H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI加密/LAI_data.xls")

"""
# 遍历文件夹对每个数据进行读取，并将结果保存到excel表格中

for root, dirs, files in os.walk(r"H:/国土卫星中心文档/内蒙野外实验数据/光谱数据/SVC数据/0724"):
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
                sheet.write(0, 0, '波长')
                sheet.write(0, 3, '反射率')
                sheet.write(0, 1, 'unknown-1')
                sheet.write(0, 2, 'unknown-2')
                i = 1
                n = 1
                for line in lines:
                    i = i + 1
                    if i > 31:
                        tmp_line = line.split('  ')
                        # 写入data
                        sheet.write(n, 0, tmp_line[0])
                        sheet.write(n, 1, tmp_line[1])
                        sheet.write(n, 2, tmp_line[2])
                        sheet.write(n, 3, tmp_line[3])
                        n = n + 1

                # 保存excel
                save_list = data_list + '.xls'
                book.save(save_list)
"""
print('finish!!!')
