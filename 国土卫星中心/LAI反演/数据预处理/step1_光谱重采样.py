import openpyxl
import numpy as np
import os
import time
import random

start_time = time.time()
a = np.loadtxt(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/1.txt")
print(a.shape)
print(a[1, 1])
print(a.dtype)

for root, dirs, files in os.walk(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI Select"):
    tmp_1 = 1
    for file in files:
        # 获取文件的路径及其文件列表
        # print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        with open(data_list, "r")as f:
            # with open(data_list, encoding='utf-8').read() as f:
            lines = f.readlines()
            i = 1
            n = 2
            ref_array = []
            for line in lines:
                i = i + 1
                if i > 2:
                    tmp_line = line.split(' ')
                    # 写入data
                    ref_tmp = tmp_line[1]
                    ref = float(ref_tmp[:-2])
                    ref=ref+random.gauss(mu=0,sigma=0.01)
                    ref_array.append(ref)
                    n = n + 1
            ref_array = np.array(ref_array)
            ref_array = ref_array.reshape(1, 2101)
            # print(ref_array.dtype)
            resample = np.matmul(ref_array, a)
            resample = resample.reshape(166, 1)
            # print(resample.shape)
            data_list = 'H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI_resample_gauss/'
            save_name = data_list + file
            np.savetxt(save_name, resample, fmt='%.18e', delimiter=' ')
end_time = time.time() - start_time
print('finish')
print(end_time)
