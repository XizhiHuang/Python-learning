"""
# round(x,d)    对x四舍五入，d是小数截取位数

# x/y 除
# x//y 整除
# x%y 取余数
# x**y x的y次幂

"""



"""
daydayup = pow(1.01, 365)
daydaydown = pow(0.99, 365)
print("daydayup:{:.2f}".format(daydayup))
print("daydaydown:{:.2f}".format(daydaydown))

# 工作五天，每天增长1%，休息两天，每天退步1%
powers = 1
for i in range(365):
    if i % 7 == 5 or i % 7 == 1 or i % 7 == 2 or i % 7 == 3 or i % 7 == 4:
        powers = powers * 1.017
    else:
        if i % 7 == 6 or i % 7 == 7:
            powers = powers * 0.99
print("result:{:.2f}".format(powers))

# A每天进步1%，B工作五天休息两天，B要多努力才能和A一样

"""
workday = 0
holiday = 0
for i in range(365):
    if i % 7 == 0 or i % 7 == 1 or i % 7 == 2 or i % 7 == 3 or i % 7 == 4:
        workday = workday + 1
    else:
        if i % 7 == 5 or i % 7 == 6:
            holiday = holiday + 1
"""

rate = 0
power = 1
while power < 37.78:
    # 每进入一次循环要对power值进行重新赋值，不然power会不断累积evale
    power = 1
    rate = rate + 0.001
    for i in range(365):
        if i % 7 == 5 or i % 7 == 1 or i % 7 == 2 or i % 7 == 3 or i % 7 == 4:
            power = power * (1 + rate)
        else:
            if i % 7 == 6 or i % 7 == 7:
                power = power * 0.99

print("result_power:{:.2f}".format(power))
print("result_rate:{:.3f}".format(rate))
"""

