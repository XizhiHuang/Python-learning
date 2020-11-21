"""

import time

scale = 10
print("------执行开始------")
for i in range(scale + 1):
    a = '*' * i
    b = '.' * (scale + 1 - i)
    c = i * 10
    print("{:^3}%[{:}->{:}]".format(c, a, b))
    time.sleep(1)
print("------执行结束------")

"""

"""

import time

for i in range(101):
    print("\r{:^3}%".format(i), end='')
    time.sleep(0.2)
for i in range(101):
    print("\r{:3}%".format(i), end='')
    time.sleep(0.2)
"""

import time

scale = 20
print("执行开始".center(scale // 2, '-'))
start = time.perf_counter()
for i in range(scale + 1):
    a = '*' * i
    b = "." * (scale - i)
    c = i * 100 / scale
    dur = time.perf_counter() - start
    print("\r{:^3}%{:}->{:}{:.2f}s".format(c, a, b, dur), end='')

    time.sleep(1)
print("执行结束".center(scale // 2, '-'))
