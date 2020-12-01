import requests
import bs4
from bs4 import BeautifulSoup


def gethtml(url):
    r = requests.get(url)
    r.encoding = r.apparent_encoding
    gethtml = r.text
    return gethtml


def fillunivlist(html, univlist):
    soup = BeautifulSoup(html, 'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr('td')
            univlist.append([tds[0].string, tds[1].string, tds[2].string, tds[3].string, tds[4].string])


def printunivlist(univlist, num):
    print('{:^5}\t{:^10}\t{:^5}\t{:^6}\t{:^8}'.format('排名', '学校名称', '省市', '类型', '总分'))
    for i in range(num):
        u = str(univlist[i])
        print('{:^5}\t{:^10}\t{:^5}\t{:^6}\t{:^8}'.format(u[0], u[1], u[2], u[3], u[4]))


url = 'https://www.shanghairanking.cn/rankings/bcur/2020'
html = gethtml(url)
univlist = []
fillunivlist(html, univlist)
printunivlist(univlist, 30)
