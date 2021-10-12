# 调用一些和操作系统相关的函数
import os
# 输入输出相关
from skimage import io
# dataset相关
import torchvision.datasets.mnist as mnist

# 路径

# root="/home/s/PycharmProjects/untitled/fashion-mnist/data/fashion"
root = "H:/Paper Code/fashion-mnist"
# 读取二进制文件，这里不知道是不是必须使用mnist读
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),  # 路径拼接，split()是分割路径与文件名，和这个正好相反
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

# 打印test_set类型
print(type(test_set))

# 打印test_set中元素个数
print(len(test_set))

# 打印第元素类型，都是tensor
print(type(test_set[0]))
print(type(test_set[1]))

# 打印元素形状,可以第一个元素是所有照片的tensor，第二个元素是所有标签的tensor.这里用test_set[0].shape是一样的


print("test set[0] :", test_set[0].size())
print("test set[1] :", test_set[1].size())

# 取出一个图片看一下，这两种都可以，就是看一下这个tensor的形状
a = test_set[0]
print(a[0].shape)
print(test_set[0][0].shape)


# 定义一个tensor转图片的函数
def convert_to_img(train=True):
    if (train):
        # 创建一个train.txt文件，用来保存标签
        f = open(root + 'train.txt', 'w')  # python中并没有这种路径表示方式，这个不对
        data_path = root + '/train/'
        # 如果不存在这个路径，就创建文件夹
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        # zip打包成元组，train_set本来不就是元组么？
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            # tensor与numpy格式转换tensor_img = torch.from_numpy(numpy_img)
            io.imsave(img_path, img.numpy())
            a = str(label)
            a = a.rstrip(')')
            a = a.strip('tensor(')  # 这里如果不进行字符串的处理，会输出“tensor(9)”而不是“９”
            f.write(img_path + ' ' + a + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            a = str(label)
            a = a.rstrip(')')
            a = a.strip('tensor(')
            f.write(img_path + ' ' + a + '\n')
        f.close()


convert_to_img(True)
convert_to_img(False)
