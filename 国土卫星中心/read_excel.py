import numpy as np
import xlrd

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
print(resArray[:,1])