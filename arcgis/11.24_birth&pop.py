import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

filepath = r'D:\作业相关\地信工程\arcpy\1.csv'

data = pd.read_csv(filepath, na_values=['0'])
df2 = data.copy()

print(data.shape)

startSeries = df2['birth year']

value_sta = pd.value_counts(startSeries).sort_index()
print(value_sta)

plt.rcParams['font.family'] = 'SimHei'

plt.bar(x=value_sta.index.to_list(), height=value_sta.values_to_list(), width=0.2, alpha=0.8, color='red', label="3")
plt.xlabel('出生日期')
plt.ylabel('单车使用次数')
plt.title('2019年6月纽约花旗单车使用记录')
