import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



filepath = r'D:\作业相关\地信工程\arcpy\201906-citibike-tripdata.csv'
outpath = r'D:/作业相关/地信工程/arcpy/'

data = pd.read_csv(filepath, na_values=['0'])
df2 = data.copy()

startSeries = df2['starttime']

times = []
for i in range(30):
    if i < 9:
        day = '0' + str(i + 1)
    else:
        day = str(i + 1)
    select_bool = startSeries.str.contains(r'2019-06-' + day, regex=False)
    selectdata = df2[select_bool]

    print(selectdata.shape[0])
    times.append(selectdata.shape[0])

    outpath_file = os.path.join(outpath, str(i + 1), '.csv')
    selectdata.to_csv(outpath_file, index=False)

print("success!")

plt.reParams['font.family'] = 'SimHei'
plt.bar(x=list(range(1, 31)), height=times, width=0.4, alpha=0.8, color='red', label='3')
plt.xlabel('日期')
plt.ylabel('单车使用记录数')
plt.title('2019年6月纽约花旗单车使用记录')