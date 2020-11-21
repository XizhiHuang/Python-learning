tempstr = input("请输入带有符号的温度值：")
# tempstr[-1]获取倒数第一个字符
if tempstr[-1] in ['F', 'f']:
    # tempstr[0:-1] 去字符串第一个到倒数第二个字符
    # 
    C = (eval(tempstr[0:-1]) - 32) / 1.8
    print("转换后的温度是：%.2f" % C)
    # print格式化输出
    print("转换后的温度是:{:.2f}C".format(C))
else:
    if tempstr[-1] in ['C', 'c']:
        F = 1.8 * (eval(tempstr[0:-1]) + 32)
        print("转换后的温度是:%.2f" % F)
    else:
        print("数据输入错误")
