# 第一位：形近距离
# 第二位：转向判断，0左转，1右转
# 第三位：转向角度
# 第四五六位：RGB通道

import turtle

turtle.title('自动绘制轨迹')
turtle.setup(700, 700, 100, 100)
turtle.pensize(5)
turtle.pencolor("red")

f = open('C:/Users/Xizhi Huang/Desktop/data.txt', 'r')
datamatrix = []
for line in f.readlines():
    line = line.replace('\n', '')
    datamatrix.append(list(map(eval, line.split(','))))
for i in range(len(datamatrix)):
    print(datamatrix[i])
for i in range(len(datamatrix)):
    turtle.fd(datamatrix[i][0])
    if datamatrix[i][1] == 0:
        turtle.left(datamatrix[i][2])
    else:
        if datamatrix[i][1] == 1:
            turtle.right(datamatrix[i][2])
    turtle.pencolor(datamatrix[i][3], datamatrix[i][4], datamatrix[i][5])
turtle.done()
f.close()