# block_copy = data_input.copy()
# block = block_copy[:, k - t_out:k + t_out + 1, j - t_out:j + t_out + 1]
# block = data_input[:, k - t_out:k + t_out + 1, j - t_out:j + t_out + 1]

import numpy as np

# 三维数组
band=134
height=2051
width=1999

win_out=11
win_in=3

t_out = int(win_out / 2)
t_in = int(win_in / 2)
m = win_out * win_out

# 一次性镜像 存为数组
# 对八边进行填充
# 形成一个band*3row*3col的数组
data_input = np.zeros((band, height+2*t_out, width+2*t_out))

data=[]

# 将原始数据填入中间
for i in range(band):
    for j in range(t_out, t_out+height):
        for k in range(t_out, t_out+width):
            data_input[i][j][k] = data[i][j - height][k - width]

# 将填入的数据向左镜像填充
for i in range(band):
    for j in range(t_out, t_out+height):
        for k in range(0, t_out):
            data_input[i][j][k] = data[i][j - t_out][t_out - k - 1]

# 将填入的数据向右镜像填充
for i in range(band):
    for j in range(t_out, t_out+height):
        for k in range(t_out+width, 2 * t_out+width):
            data_input[i][j][k] = data[i][j - t_out][2 * t_out+width - k - 1]

# 将上面填充的结果整体向上翻转镜像填充
for i in range(band):
    for j in range(0, t_out):
        for k in range(0, 2 * t_out+width):
            data_input[i][j][k] = data_input[i][t_out+height - j - 1][k]

# 将上面填充的结果整体向下翻转镜像填充
for i in range(band):
    for j in range(t_out+height, 2 * t_out+height):
        for k in range(0, 2 * t_out+width):
            data_input[i][j][k] = data_input[i][ t_out+2*height - j - 1][k]





# 创建四个角落需要两侧均镜像操作的数组 将其进行保存，反复调用
left_top_array=[]
right_top_array=[]
left_under_array=[]
right_under_array=[]




# 对于四边的仅需要单侧镜像操作的区域

for i in range(band):
    for j in range(height):
        for k in range(width):
            # 划分成九宫格，中间直接执行block操作，其余八个部分根据不同的情况做镜像操作
            # 左上角两侧同时镜像操作
            if j<t_out and k<t_out:
