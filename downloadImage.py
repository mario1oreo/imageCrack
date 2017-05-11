# import requests
from urllib import request


def downloads_pic(**kwargs):
    pic_name = kwargs.get('pic_name', None)
    pic_path = kwargs.get('pic_path', None)
    url = 'http://cpquery.sipo.gov.cn/freeze.main?txn-code=createImgServlet&freshStept=1&now=2323123'
    res = request.urlopen(url)
    with open(pic_path + pic_name + '.bmp', 'wb') as f:
        f.write(res.read())
        f.flush()
        f.close()


for i in range(501, 503):
    downloads_pic(pic_name=str(i), pic_path='c:/image/pythonImage/testImage/')
