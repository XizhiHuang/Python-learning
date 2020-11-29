# PythonDraw.py
# turtle绘图库
# import turtle

"""
turtle.setup(width,height,startx,starty)
width,height表示实际框口的大小
startx,starty表示相对于电脑屏幕的位置 以电脑左上角为起始点

# turtle绝对坐标系
1、以起始点构造平面直角坐标系
2、goto函数
import turtle
turtle.goto(100,100)
turtle.goto(100,-100)
turtle.goto(-100,-100)
turtle.goto(-100,100)
turtle.goto(0,0)

# turtle海龟坐标
1、以海龟当前的行进方向来确定其坐标
2、相关函数
turtle.circle(r,range)  从当前点开始以左侧某一点为圆心进行曲线运动
turtle.fd(d)            从当前点正前方方向运动
turtle.bd(d)            从当前点正后方方向运动

# turtle 角度坐标体系
1、东0°或360°，北90°或-270°，西180°或-180°，南270°或-90°
2、相关函数
turtle.seth(angle)
3、绘制Z曲线
import turtle
# turtle.setup(400,400,400,400)
turtle.left(45)         控制角度
turtle.fd(150)          控制移动距离
turtle.right(135)
turtle.fd(300)
turtle.left(135)
turtle.fd(135)
turtle.done()           控制函数结束

# turtle RGB色彩模式
# 常见RGB色彩
  white     255,255,255
  yellow    255,255,0
  magenta   255，0,255
  cyan      0,255,255
  blue      0,0,255
  black     0,0,0
turtle.colormode(mode)


from turtle import *
setup(650, 350, 200, 200)
penup()
fd(-250)
pendown()
pensize(25)
pencolor("red")
seth(-40)
for i in range(4):
    circle(40, 80)
    circle(-40, 80)
circle(40, 80 / 2)
fd(40)
circle(16, 180)
fd(40 * 2 / 3)
done()

# 导入函数库的两种方法
1、import xxx as x
import turtle as t
t.setup(650, 350, 200, 200)
t.penup()
t.fd(-250)
t.pendown()
t.pensize(25)
t.pencolor("red")
t.seth(-40)
for i in range(4):
    t.circle(40, 80)
    t.circle(-40, 80)
t.circle(40, 80 / 2)
t.fd(40)
t.circle(16, 180)
t.fd(40 * 2 / 3)
t.done()

2、from xxx import *
from turtle import *
setup(650, 350, 200, 200)
penup()
fd(-250)
pendown()
pensize(25)
pencolor("red")
seth(-40)
for i in range(4):
    circle(40, 80)
    circle(-40, 80)
circle(40, 80 / 2)
fd(40)
circle(16, 180)
fd(40 * 2 / 3)
done()
"""

import turtle

turtle.setup(650, 350, 200, 200)
# 画笔控制函数 penup和pendown成双成对出现
# turtle.penup()和turtle.pu() 拾起画笔
# turtle.pendown()和turtle.pd() 落下画笔
# turtle.pensize(width)turtle.width(width) 画笔宽度
# turtle.pencolor(color) 画笔颜色 color为颜色字符或r，g，b值
# turtle.pencolor("purple")             颜色字符串
# turtle.pencolor(0.63,0.13,0.94)       RGB的小数值
# turtle.pencolor((0.63,0.13,0.94))     RGB的元组值
turtle.penup()
turtle.fd(-250)
turtle.pendown()
turtle.pensize(25)
turtle.pencolor("red")
# turtle.seth 绝对坐标系
turtle.seth(-40)
for i in range(4):
    # turtle.circle(radius,extent)
    # radius为正数，圆心在小海龟左侧，反之在右侧
    # extent为正数，顺小海龟当前方向绘制，反之逆向
    turtle.circle(40, 80)
    turtle.circle(-40, 80)
turtle.circle(40, 80 / 2)
turtle.fd(40)
turtle.circle(16, 180)
turtle.fd(40 * 2 / 3)
turtle.done()


