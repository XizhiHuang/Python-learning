from PIL import Image as img
import os
from tqdm import tqdm

in_root = r'C:/Users/Xizhi Huang/Desktop/clip/raw/'  # 初始图像存放文件夹
out_root = r'C:/Users/Xizhi Huang/Desktop/clip/clip/'  # 处理后图片保存文件夹

rootdirl = in_root
file_list = os.listdir(rootdirl)
a = 0
for ii, files in tqdm(enumerate(file_list), total=len(file_list)):
    imagepath = os.path.join(rootdirl, files)  # 生成每个图片的路径
    jpg = img.open(imagepath)
    """
    box_list中的每个元组四个元素指定两个坐标点，分别为切割部分的左上角和右下角的坐标，坐标数值为沿x轴或y轴到图案左上角的距离，单位为像素点
    切割时每次保留正中间的那一小块图案，如果从正中间十字切开，会导致图片主体被割裂
    """
    box_list = [(750, 250, 1750, 1250), (1250, 250, 2250, 1250),
                (750, 750, 1750, 1750), (1250, 750, 2250, 1750)]

    jpg_list = [jpg.crop(box) for box in box_list]
    for image in jpg_list:
        image = image.resize((480, 480), img.ANTIALIAS)  # 可根据个人需要设置resize大小
        a += 1
        image.save("%s/%d.jpg" % (out_root, a))
