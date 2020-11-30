import requests

r = requests.get("http://www.taobao.com")
print(r.status_code)
r.encoding = 'utf-8'
print(r.text)
