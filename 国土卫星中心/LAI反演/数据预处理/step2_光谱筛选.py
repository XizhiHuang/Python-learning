import shutil
import os

# 对重采样后的光谱筛选
# 对特定LAI进行筛选
for root, dirs, files in os.walk(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI_resample_gauss"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        #print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        data_name = data_list[-25:]
        #print(data_name)
        if data_list[58:61] == '2.0':
            # shutil.copy("file.txt", "file_copy.txt")
            data_copy_list = "H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI_resample_gauss/2.0/" + data_name
            shutil.copy(data_list, data_copy_list)
            #print('copy')

print('finish!!!')