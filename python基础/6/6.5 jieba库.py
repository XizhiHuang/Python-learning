import jieba

s = jieba.lcut('中国是一个伟大的国家')
for i in range(len(s)):
    print(s[i])
