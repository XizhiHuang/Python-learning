# 递归--汉诺塔

"""

从A盘借助B盘移动到C盘
move函数：定义移动
hanoi函数：执行迭代

"""


def move(get, to):
    print("{:}-->{:}".format(get, to))


def hanoi(n, get, med, to):
    if n == 1:
        move(get, to)
    else:
        hanoi(n - 1, get, to, med)
        print("{:}-->{:}".format(get, to))
        hanoi(n - 1, med, get, to)


hanoi(4, 'A', 'B', 'C')
