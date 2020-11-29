# 体育竞技分析
# step1：打印程序的介绍性信息              printinfo
# step2：获得程序运行参数proA，proB,n      getinput
# step3：获得球员能力值，模拟比赛           simgame
# step4：输出获胜场次信息及其概率           printresult

# 比赛五局三胜 每一局谁先得到15分获胜

def main():
    printinfo()
    proA, proB, n = getinput()
    simgame()
    printresult()


def getinput():
    proA = input("please enter A's ablity:")
    proB = input("please enter B's ablity:")
    n = input("please enter the number of match:")
    return proA, proB, n

def printinfo():
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('')