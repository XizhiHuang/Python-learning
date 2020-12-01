# https://python123.io/ws/demo.html

import requests
from bs4 import BeautifulSoup

r = requests.get('https://python123.io/ws/demo.html')
demo = r.text
soup = BeautifulSoup(demo, 'html.parser')
"""
print(soup.title.string)
print(soup.a)
print(soup.a.name)
print(soup.a.parent.name)
print(soup.a.attrs)
print(soup.a.attrs['class'])


print(soup.head)
print(soup.head.contents)
print(soup.body.contents)
print(len(soup.body.contents))
for i in range(len(soup.body.contents)):
    print(soup.body.contents[i])
"""

"""

for parent in soup.a.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)


print(soup.a.next_sibling)
print(soup.a.next_sibling.next_sibling)
print(soup.a.previous_sibling)

"""

print(soup.prettify())

for link in soup.find_all('a'):
    print(link.get('class'))
    print(link.get('href'))

# 引入正则化库re，获得所有以b开头的标签
import re

for tag in soup.find_all(re.compile('b')):
    print(tag.name)

print(soup.find_all(id='link1'))
