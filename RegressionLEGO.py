"""
1.预测乐高玩具套装的价格
2.使用Google购物的API
3.
姓名：pcb
日期：2019.1.6
"""

import urllib2
from time import sleep
import json

#调用Google的API，并保证数据抽取的正确性
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr='get from code.google.com'
    searchURL='https://www.googleapis.com/shopping/search/v1/public/products?' \
              'key=%s&country=US&q=lego+%d&alt=json'%(myAPIstr,setNum)
