"""

f = open('f.txt')
f.readline()
f.close()


"""

"""

fname = input('please enter name:')
fo = open(fname, 'r')
for line in fo.readlines():
    print(line)
fo.close()
"""

fname = input('please enter name:')
fo = open(fname, 'a+')
insert = ['中国', '人民', '共和国', '哈哈哈哈']
fo.writelines(insert)
fo.seek(0)
for line in fo.readlines():
    print(line)
fo.close()
