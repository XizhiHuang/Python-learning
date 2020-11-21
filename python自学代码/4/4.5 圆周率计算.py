# 圆周率的计算 使用公式

N = 100
pi = 0
for k in range(N):
    pi = pi + (1 / pow(16, k)) * ((4 / (8 * k + 1)) - (2 / (8 * k + 4)) -
                                  (1 / (8 * k + 5)) - (1 / (8 * k + 6)))
print("圆周率的值为：{:}".format(pi))

# 使用随机函数的方式计算
import random
import time

size = 5000 * 5000
point = 0
pi_random = 0
start = time.perf_counter()
for i in range(1, size + 1):
    r = 0
    x, y = random.random(), random.random()
    r = pow((x * x + y * y), 0.5)
    if r <= 1:
        point = point + 1
pi_random = 4 * point / size
end = time.perf_counter() - start
print("圆周率的值为：{:}".format(pi_random))
print("耗时：{:}".format(end))
