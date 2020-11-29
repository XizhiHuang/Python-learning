import jieba
import wordcloud
from imageio import imread

# 掩膜提取好像用不了？？？？
"""
mask = imread('C:/Users/Xizhi Huang/Desktop/fivestar.png')
"""

f_gov = open('C:/Users/Xizhi Huang/Desktop/实施乡村振兴战略的意见.txt', 'r', encoding='utf-8')
t = f_gov.read()
f_gov.close()
ls_gov = jieba.lcut(t)
txt = ''.join(ls_gov)
w = wordcloud.WordCloud(font_path='msyh.ttc', width=1000, height=1000,
                        background_color='white') #mask=mask)
w.generate(txt)
w.to_file('C:/Users/Xizhi Huang/Desktop/country.jpg')
