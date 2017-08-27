# coding=utf-8
# 对图片进行一些灰度化  二值化相关的处理
import os
from PIL import Image


def initTable():
    table = []
    for i in range(256):
        if i < 235:
            table.append(0)
        else:
            table.append(1)
    return table


if __name__ == '__main__':
    # imgPath = 'D:/work/captcha/genCaptcha/originalImage/'
    # dealImg = 'D:/work/captcha/genCaptcha/originalImageTemp/'
    imgPath = 'D:/work/captcha/genCaptcha/test/'
    dealImg = 'D:/work/captcha/genCaptcha/testTemp/'
    if os.path.isdir(imgPath):
        for f in os.listdir(imgPath):
            if f.split('.')[-1] in ['bmp', 'jpeg', 'gif', 'psd', 'png', 'jpg']:
                print(f)
                im = Image.open(imgPath + f)
                im = im.convert('L')
                binaryImage = im.point(initTable(), '1')
                region = (0, 0, 160, 60)
                img = binaryImage.crop(region)
                # binaryImage.show()
                img.save(dealImg + f)
