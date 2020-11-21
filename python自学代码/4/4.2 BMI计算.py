# 输入身高体重 计算得到国内和国外两套标准结果


# 国内
import time

height = eval(input("请输入你的身高（米）："))
weight = eval(input("请输入你的体重（公斤）："))
start = time.perf_counter()
BMI = weight / pow(height, 2)
print("BMI数值为：{:.2f}".format(BMI))
level_globe = ''
level_china = ''
if BMI < 18.5:
    level_globe = '偏瘦'
    level_china = '偏瘦'
else:
    if 18.5 <= BMI < 24:
        level_globe = '正常'
        level_china = '正常'
    else:
        if 24 <= BMI < 25:
            level_globe = '正常'
            level_china = '偏胖'
        else:
            if 25 <= BMI < 28:
                level_globe = '偏胖'
                level_china = '偏胖'
            else:
                if 28 <= BMI < 30:
                    level_globe = '偏胖'
                    level_china = '肥胖'
                else:
                    level_globe = '肥胖'
                    level_china = '肥胖'

print("BMI国际标准为；{:}\nBMI国内标准为；{:}".format(level_globe, level_china))
end = time.perf_counter() - start
print(end)
