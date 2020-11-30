"""
import requests

kv = {'user-agent': 'Mozilla/5.0'}
url = "https://www.amazon.cn/dp/B0118GQM1A?_encoding=UTF8&ref_=pc_cxrd_2045366051_recTab_2045366051_t_1&pf_rd_p=577c845d-13b2-4e81-8a3e-772c4d55db4b&pf_rd_s=merchandised-search-4&pf_rd_t=101&pf_rd_i=2045366051&pf_rd_m=A1AJ19PSB66TGU&pf_rd_r=TPADHK798MN5J65J0KKR&pf_rd_r=TPADHK798MN5J65J0KKR&pf_rd_p=577c845d-13b2-4e81-8a3e-772c4d55db4b"
r = requests.get(url, headers=kv)
print(r.status_code)
print(r.request.headers)
r.raise_for_status()


r.encoding = r.apparent_encoding
print(r.text[1000:2000])

"""

"""
import requests

kv = {'wd': 'python'}
url = 'http://www.baidu.com/s'
r = requests.get(url, params=kv)
print(r.status_code)
print(r.request.url)
"""



"""
import requests
import os

url = 'https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=1767516822,2482793911&fm=26&gp=0.jpg'
root = 'C:/Users/Xizhi Huang/Desktop/'
path = root + url.split('/')[-1]
path = root + 'ts.jpg'
if not os.path.exists(root):
    os.mkdir(root)
if not os.path.exists(path):
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
        f.close()
        print("yes")
else:
    print("文件已经存在！")

"""



"""

# 网站有问题
# https://www.ip138.com/iplookup.asp?ip=219.142.99.9&action=2
# https://www.ip138.com/iplookup.asp?ip=219.142.99.8&action=2

import requests

#kv = {'ip': '219.142.99.9''&action=2'}
url = 'https://www.ip138.com/iplookup.asp?ip='
#r = requests.get(url, params=kv)
r = requests.get(url + '219.142.99.9'+'&action=2')
print(r.status_code)

"""







