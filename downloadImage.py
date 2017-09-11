# import requests
from urllib import request

import time


def downloads_pic(**kwargs):
    pic_name = str(kwargs.get('pic_name', None))
    pic_path = kwargs.get('pic_path', None)
    # url = 'http://cpquery.sipo.gov.cn/freeze.main?txn-code=createImgServlet&freshStept=1&now=2323123'

    url = 'http://101.201.34.58/wsfy-ww/cpws_yzm.jpg?n=1'
    res = request.urlopen(url)
    with open(pic_path + pic_name + '.bmp', 'wb') as f:
        f.write(res.read())
        f.flush()
        f.close()
        print(pic_path + pic_name + '.bmp')
    time.sleep(1)


for i in range(4217, 6000):
    downloads_pic(pic_name=i, pic_path='D:/work/captcha/susongwuyou/')
