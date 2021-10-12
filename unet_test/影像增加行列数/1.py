
"""
import os

json_folder = r"D:/result/1/label_json/"
#  获取文件夹内的文件名
FileNameList = os.listdir(json_folder)
#  激活labelme环境
os.system("activate labelme")
for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if(os.path.splitext(FileNameList[i])[1] == ".json"):
        json_file = json_folder + "\\" + FileNameList[i]
        #  将该json文件转为png
        os.system("labelme_json_to_dataset " + json_file)
"""



import os
import shutil

JPG_folder = r"D:/result/1/test/label_json/"
Paste_JPG_folder = r"D:/result/1/test/png/"
Paste_label_folder = r"D:/result/1/test/label/"
#  获取文件夹内的文件名
FileNameList = os.listdir(JPG_folder)
NewFileName = 1
for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if(os.path.splitext(FileNameList[i])[1] == ".json"):

        #  复制jpg文件
        JPG_file = JPG_folder + "\\" + FileNameList[i]
        new_JPG_file = Paste_JPG_folder + "\\" + str(NewFileName) + ".png"
        shutil.copyfile(JPG_file, new_JPG_file)

        #  复制label文件
        jpg_file_name = FileNameList[i].split(".", 1)[0]
        label_file = JPG_folder + "\\" + jpg_file_name + "_json\\label.png"
        new_label_file = Paste_label_folder + "\\" + str(NewFileName) + ".png"
        shutil.copyfile(label_file, new_label_file)

        #  文件序列名+1
        NewFileName = NewFileName + 1