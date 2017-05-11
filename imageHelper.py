# coding=utf-8
# 验证码识别，此程序只能识别数据验证码
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import sys
import os
import stat
from svmutil import *
import uuid

# from pytesser import *

base_path = 'pythonImage/result/'


# 二值化
def get_bin_table():
    threshold = 140
    table = []
    for ii in range(256):
        if ii < threshold:
            table.append(0)
        else:
            table.append(1)
    return table


# 灰度化
def toGrey(im):
    imgry = im.convert('L')  # 转化为灰度图
    table = get_bin_table()
    out = imgry.point(table, '1')
    return out


# 黑点个数
def sum_9_region(img, x, y):
    width = img.width
    height = img.height
    flag = getflag(img, x, y)
    # 如果当前点为白色区域,则不统计邻域值
    if flag == 0:
        return 0
    # 如果是黑点
    if y == 0:  # 第一行
        if x == 0:  # 左上顶点,4邻域
            # 中心点旁边3个点
            total = getflag(img, x, y + 1) + getflag(img, x + 1, y) + getflag(img, x + 1, y + 1)
            return total
        elif x == width - 1:  # 右上顶点
            total = getflag(img, x, y + 1) + getflag(img, x - 1, y) + getflag(img, x - 1, y + 1)
            return total
        else:  # 最上非顶点,6邻域
            total = getflag(img, x - 1, y) + getflag(img, x - 1, y + 1) + getflag(img, x, y + 1) \
                    + getflag(img, x + 1, y) \
                    + getflag(img, x + 1, y + 1)
            return total
    elif y == height - 1:  # 最下面一行
        if x == 0:  # 左下顶点
            # 中心点旁边3个点
            total = getflag(img, x + 1, y) + getflag(img, x + 1, y - 1) + getflag(img, x, y - 1)
            return total
        elif x == width - 1:  # 右下顶点
            total = getflag(img, x, y - 1) + getflag(img, x - 1, y) + getflag(img, x - 1, y - 1)
            return total
        else:  # 最下非顶点,6邻域
            total = getflag(img, x - 1, y) + getflag(img, x + 1, y) + getflag(img, x, y - 1) + getflag(img, x - 1,
                                                                                                       y - 1) + getflag(
                img, x + 1, y - 1)
            return total
    else:  # y不在边界
        if x == 0:  # 左边非顶点
            total = getflag(img, x, y - 1) + getflag(img, x, y + 1) + getflag(img, x + 1, y - 1) + getflag(img, x + 1,
                                                                                                           y) + getflag(
                img, x + 1, y + 1)
            return total
        elif x == width - 1:  # 右边非顶点
            total = getflag(img, x, y - 1) + getflag(img, x, y + 1) + getflag(img, x - 1, y - 1) + getflag(img, x - 1,
                                                                                                           y) + getflag(
                img, x - 1, y + 1)
            return total
        else:  # 具备9领域条件的
            total = getflag(img, x - 1, y - 1) + getflag(img, x - 1, y) + getflag(img, x - 1, y + 1) + getflag(img, x,
                                                                                                               y - 1) \
                    + getflag(img, x, y + 1) + getflag(img, x + 1, y - 1) + getflag(img, x + 1, y) + getflag(img, x + 1,
                                                                                                             y + 1)
            return total


# 判断像素点是黑点还是白点
def getflag(img, x, y):
    tmp_pixel = img.getpixel((x, y))
    if tmp_pixel > 228:  # 白点
        tmp_pixel = 0
    else:  # 黑点
        tmp_pixel = 1
    return tmp_pixel


# 去除噪点
def greyimg(image):
    width = image.width
    height = image.height
    box = (0, 0, width, height)
    imgnew = image.crop(box)
    for i in range(0, height):
        for j in range(0, width):
            num = sum_9_region(image, j, i)
            if num < 2:
                imgnew.putpixel((j, i), 255)  # 设置为白色
            else:
                imgnew.putpixel((j, i), 0)  # 设置为黑色
    return imgnew


# 图片像素70*20
# 每个字符间隔9像素
# 每个字符宽8个字符
# 字符的外边距,上下为5，左右为6. 分割后查看效果，然后适当的优化一下，最后的源码如下：
# 分割图片
def spiltimg(img):
    # 按照图片的特点,进行切割,这个要根据具体的验证码来进行工作.
    child_img_list = []
    for index in range(4):
        x = 3 + index * (10 + 8)
        y = 3
        # img.show()
        child_img = img.crop((x, y, x + 10, img.height - 4))
        # child_img.show()
        child_img_list.append(child_img)
    return child_img_list


def get_feature(img):
    # 获取指定图片的特征值,
    # 1. 按照每排的像素点,高度为12,则有12个维度,然后为8列,总共20个维度
    # :return:一个维度为20（高度）的列表
    width, height = img.size
    pixel_cnt_list = []
    for y in range(height):
        pix_cnt_x = 0
        for x in range(width):
            if img.getpixel((x, y)) <= 100:  # 黑色点
                pix_cnt_x += 1
        pixel_cnt_list.append(pix_cnt_x)
    for x in range(width):
        pix_cnt_y = 0
        for y in range(height):
            if img.getpixel((x, y)) <= 100:  # 黑色点
                pix_cnt_y += 1
        pixel_cnt_list.append(pix_cnt_y)

    return pixel_cnt_list


def train(filename, merge_pic_path):
    if os.path.exists(filename):
        os.remove(filename)
    result = open(filename, 'a')
    for f in os.listdir(merge_pic_path):
        if f != '.DS_Store' and os.path.isdir(merge_pic_path + f):
            for img in os.listdir(merge_pic_path + f):
                if img.endswith(".bmp"):
                    pic = Image.open(merge_pic_path + f + "/" + img)
                    pixel_cnt_list = get_feature(pic)
                    # if ord(f) > 9:
                    # 所有字符都转码
                    line = str(ord(f)) + " "
                    # else:
                    #     line = f + " "
                    for i in range(1, len(pixel_cnt_list) + 1):
                        line += "%d:%d " % (i, pixel_cnt_list[i - 1])
                    print('line:', line)
                    result.write(line + "\n")
    result.close()


# 模型训练
def train_svm_model(filename):
    y, x = svm_read_problem(base_path + filename)
    model = svm_train(y, x)
    svm_save_model(base_path + "svm_model_file", model)


def train_new(filename, path_new):
    if os.path.exists(filename):
        os.remove(filename)
    result_new = open(filename, 'a')
    for f in os.listdir(path_new):
        if f != '.DS_Store' and f.endswith(".bmp"):
            pic = Image.open(path_new + f)
            pixel_cnt_list = get_feature(pic)
            line = "0 "
            for i in range(1, len(pixel_cnt_list) + 1):
                line += "%d:%d " % (i, pixel_cnt_list[i - 1])
            result_new.write(line + "\n")
    result_new.close()


# 使用完整图片作为测试集测试模型
def get_feature_image_file(filePath):
    if os.path.isdir(filePath):
        for f in os.listdir(filePath):
            test_date = ''
            if os.path.isdir(filePath + f):
                for fi in os.listdir(filePath + f):
                    if fi.endswith(".bmp"):
                        pic = Image.open(filePath + f + '/' + fi)
                        pixel_cnt_list = get_feature(pic)
                        line = "0 "
                        for i in range(1, len(pixel_cnt_list) + 1):
                            line += "%d:%d " % (i, pixel_cnt_list[i - 1])
                        test_date += line + '\n'
                        print('*' * 40)
                        print(test_date)
            else:
                if f.split('.')[-1] in ['bmp', 'jpeg', 'gif', 'psd', 'png', 'jpg']:
                    # 开始处理图片
                    pic = Image.open(filePath + f)
                    pic = toGrey(pic)
                    pic.save("new_code.jpg")
                    pic = Image.open("new_code.jpg")
                    newpic = greyimg(pic)
                    childs = spiltimg(newpic)
                    count = 0
                    for c in childs:
                        pixel_cnt_list = get_feature(c)
                        line = "0 "
                        for i in range(1, len(pixel_cnt_list) + 1):
                            line += "%d:%d " % (i, pixel_cnt_list[i - 1])
                        test_date += line + '\n'
                    print('*' * 40)
                    print(test_date)
            if len(test_date) == 0:
                continue
            temp_file_path = create_write_content(filePath + f + "_", test_date)
            value = svm_model_test(temp_file_path)
            if os.path.exists(temp_file_path + value[0]):
                os.remove(temp_file_path + value[0])
            os.rename(temp_file_path, temp_file_path + value[0])
        else:
            print('filePath:{%s} finished!!' % filePath)
    else:
        print('filePath:{%s} is not a file dir!!' % filePath)


# callback 时调用删除特征文件
def del_feature_file(featureFilePath):
    if os.path.isfile(featureFilePath):
        os.remove(featureFilePath)


def create_write_content(gener_path, content):
    tempFileName = gener_path  # + str(uuid.uuid1())
    temp_File = open(tempFileName, 'a+')
    temp_File.write(content)
    temp_File.flush()
    return tempFileName


def get_result_from_full_image(image_path):
    # 1.get feature file
    if os.path.isfile(image_path):
        if image_path.split('.')[-1] in ['bmp', 'jpeg', 'gif', 'psd', 'png', 'jpg']:
            test_date = ""
            # 开始处理图片
            pic = Image.open(image_path)
            pic = toGrey(pic)
            pic.save("new_code.jpg")
            pic = Image.open("new_code.jpg")
            newpic = greyimg(pic)
            childs = spiltimg(newpic)
            count = 0
            for c in childs:
                pixel_cnt_list = get_feature(c)
                line = "0 "
                for i in range(1, len(pixel_cnt_list) + 1):
                    line += "%d:%d " % (i, pixel_cnt_list[i - 1])
                test_date += line + '\n'
            print('*' * 40)
            print(test_date)
            create_write_content("tempFeatureFile", test_date)
    # 2.svm_model_test 获取结果
    result = svm_model_test("tempFeatureFile")
    del_feature_file("tempFeatureFile")
    print (result)


# 使用测试集测试模型
def svm_model_test(filename):
    # yt, xt = svm_read_problem(base_path + '/' + filename)
    yt, xt = svm_read_problem(filename)
    model = svm_load_model(base_path + "svm_model_file")
    p_label, p_acc, p_val = svm_predict(yt, xt, model)  # p_label即为识别的结果
    cnt = 0
    results = []
    result = ''
    for item in p_label:  # item:float
        # if int(item) > 9:
        # 所有都转回来
        result += chr(int(item))
        # else:
        # result += str(int(item))
        cnt += 1
        if cnt % 4 == 0:
            results.append(result)
            result = ''
    return results


# 批量更新
def read_files(filePath):
    if os.path.isdir(filePath):
        if not os.path.isdir(filePath + 'total'):
            # os.mkdir(filePath + 'total')
            os.makedirs(filePath + 'total')
        if os.path.isfile(filePath + 'total/feature'):
            os.remove(filePath + 'total/feature')
        total = open(filePath + 'total/feature', 'a+')
        for f in os.listdir(filePath):
            if not os.path.isfile(filePath + f):
                continue
            total.write(read_file_replace_first_key(filePath + f))
            total.flush()
        total.close()


# 更新单个文件特征值的标记
def read_file_replace_first_key(filePath):
    file = open(filePath, 'r')
    lines = file.readlines()
    newContent = ''
    count = 0
    value = filePath.split('_')[-1]
    for line in lines:
        # print(line)
        # print(value[count])
        if line == '\n':
            continue
        newContent += value[count] + line[1:]
        count += 1
    print (newContent)
    return newContent


# 验证码灰度化并且分割图片

# param 待分割图片入境  分割后图片路径

def begin(pic_path, split_pic_path):
    for f in os.listdir(pic_path):
        if os.path.isfile(pic_path + f):
            if f.endswith(".bmp"):
                pic = Image.open(pic_path + f)
                pic = toGrey(pic)
                pic.save("new_code.jpg")
                pic = Image.open("new_code.jpg")
                newpic = greyimg(pic)
                childs = spiltimg(newpic)
                count = 0
                for c in childs:
                    c.save(split_pic_path + f.split(".")[0] + "-" + str(count) + '.bmp')
                    count += 1


# begin('pythonImage/result/','pythonImage/result/')
# print(get_feature(Image.open('pythonImage/result/9/1-0.bmp')))


# train('pythonImage/result/result04182330', 'pythonImage/result/')
# train_svm_model('result04182330')

# train('pythonImage/result/test04182330', 'pythonImage/result/spilt/')
# print(svm_model_test('result04182330'))
#


# print (get_feature_image_file('pythonImage/testImage/'))


# read_file_replace_first_key('pythonImage/testImage/right/401.bmp_2+9=')
# read_files('pythonImage/testImage/right/')

#获取结果
print (get_result_from_full_image("pythonImage/testImage/401.bmp"))
