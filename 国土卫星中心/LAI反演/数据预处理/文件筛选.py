import shutil
import os

"""
for root, dirs, files in os.walk(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI加密"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        data_name = data_list[-25:]
        print(data_name)
        if data_list[45:48] == '0.3' or data_list[45:48] == '0.4' or data_list[45:48] == '0.5' or \
                data_list[45:48] == '0.6' or data_list[45:48] == '0.7' or data_list[45:48] == '0.8' or \
                data_list[45:48] == '0.9' or data_list[45:48] == '1.0' or data_list[45:48] == '1.1' or \
                data_list[45:48] == '1.2' or data_list[45:48] == '1.3' or data_list[45:48] == '1.4' or \
                data_list[45:48] == '1.5' or data_list[45:48] == '1.6' or data_list[45:48] == '1.7' or \
                data_list[45:48] == '1.8' or data_list[45:48] == '1.9' or data_list[45:48] == '2.0':
            # shutil.copy("file.txt", "file_copy.txt")
            data_copy_list = "H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI Select/" + data_name
            shutil.copy(data_list, data_copy_list)
            print('copy')

print('finish!!!')
"""

"""
# 对特定LAI进行筛选
for root, dirs, files in os.walk(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI Select"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        data_name = data_list[-25:]
        print(data_name)
        if data_list[50:53] == '2.0':
            # shutil.copy("file.txt", "file_copy.txt")
            data_copy_list = "H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI/2.0/" + data_name
            shutil.copy(data_list, data_copy_list)
            print('copy')

print('finish!!!')
"""

# 对重采样后的光谱筛选
# 对特定LAI进行筛选
for root, dirs, files in os.walk(r"H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI_resample_gauss"):
    for file in files:
        # 获取文件所属目录
        # print(root)
        # 获取文件路径
        print(os.path.join(root, file))
        data_list = os.path.join(root, file)
        data_name = data_list[-25:]
        print(data_name)
        if data_list[52:55] == '2.0':
            # shutil.copy("file.txt", "file_copy.txt")
            data_copy_list = "H:/国土卫星中心文档/LAI反演/光谱数据/LAI加密/LAI_resample_gauss/2.0/" + data_name
            shutil.copy(data_list, data_copy_list)
            print('copy')

print('finish!!!')