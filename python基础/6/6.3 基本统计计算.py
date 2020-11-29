# 计算平均数 中位数 方差

# 获得输入数字
def getnumber():
    nums = []
    numstr = input("请输入数字（回车结束）：")
    while numstr != '':
        nums.append(eval(numstr))
        numstr = input("请输入数字（回车结束）：")
    return nums


# 计算平均数
def average(nums):
    sum = 0
    for i in range(len(nums)):
        sum = sum + nums[i]
    average = sum / len(nums)
    return average


# 计算中位数
def midnum(nums):
    sorted(nums)
    if len(nums) % 2 == 0:
        midnum = (nums[(len(nums) // 2 - 1)] + nums[len(nums) // 2]) / 2
    else:
        midnum = nums[(len(nums) // 2)]
    return midnum


# 计算方差
def dev(nums):
    dev_sum = 0
    average_num = average(nums)
    for i in range(len(nums)):
        dev_sum = dev_sum + pow(nums[i] - average_num, 2)
    dev = dev_sum / len(nums)
    return dev


# 结果
n = getnumber()
print('平均数：{:}'.format(average(n)))
print('中位数：{:}'.format(midnum(n)))
print('方差：{:}'.format(dev(n)))
