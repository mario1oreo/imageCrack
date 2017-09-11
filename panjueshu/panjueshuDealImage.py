# coding=utf-8
# 对图片进行一些灰度化  二值化相关的处理
from PIL import Image


def initTable():
    table = []
    for i in range(256):
        if i < 180:
            table.append(0)
        else:
            table.append(1)
    return table


for i in range(1, 2000):
    im = Image.open('C:/image/panjueshu/originalImage/' + str(i) + '.bmp')
    im = im.convert('L')
    binaryImage = im.point(initTable(), '1')
    region = (1,1,63,21)
    img = binaryImage.crop(region)
    # binaryImage.show()
    img.save('C:/image/panjueshu/originalImageTemp1/' + str(i) + '.bmp')


