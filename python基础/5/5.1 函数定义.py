# 七段数码管绘制
import turtle
import time


# 绘制间隔
def drawgap():
    turtle.penup()
    turtle.fd(5)


# 绘制单段数码管
def drawsingle(stage):
    drawgap()
    turtle.pendown()
    if stage != 'draw':
        turtle.penup()
    turtle.fd(40)
    drawgap()
    turtle.right(90)


# 绘制七段数码管
def drawmulti(num):
    if num in [2, 3, 4, 5, 6, 8, 9]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    if num in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    if num in [0, 2, 3, 5, 6, 8, 9]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    if num in [0, 2, 6, 8]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    turtle.left(90)
    if num in [0, 4, 5, 6, 8, 9]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    if num in [0, 2, 3, 5, 6, 7, 8, 9]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    if num in [0, 1, 2, 3, 4, 7, 8, 9]:
        drawsingle('draw')
    else:
        drawsingle('notdraw')
    turtle.left(180)
    turtle.penup()
    turtle.fd(30)


def drawdate(date):
    turtle.color('purple')
    for i in date:
        if i == '-':
            turtle.write('年', font=('Arial', 22, 'normal'))
            turtle.color('pink')
            turtle.penup()
            turtle.fd(50)
        elif i == '=':

            turtle.write('月', font=('Arial', 22, 'normal'))
            turtle.color('green')
            turtle.penup()
            turtle.fd(50)
        elif i == '+':

            turtle.write('日', font=('Arial', 22, 'normal'))
            turtle.color('blue')
            turtle.penup()
            turtle.fd(50)
        else:
            drawmulti(eval(i))


turtle.setup(1000, 500, 200, 200)
turtle.penup()
turtle.fd(-400)
turtle.pensize(5)
drawdate(time.strftime('%Y-%m=%d+', time.gmtime()))
turtle.hideturtle()
turtle.done()
