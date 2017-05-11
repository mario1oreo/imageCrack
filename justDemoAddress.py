# coding = utf-8
import re

file = open('e:/demo.txt')
lines = file.readlines()
print(lines)

dateflag = False
for line in lines:
    print('*' * 40)
    if line == '\n':
        print('当前是换行符，下一行是日期')
        dateflag = True
        continue
    if dateflag and line != '\n':
        print('当前行是工作日期！')
        print('content:' + line)
        dateflag = False
    else:
        print('当前行是工作内容！')
        print('content:' + line)
        allCont = line.split(' ')

        print(allCont[0])
        print(allCont[1])
        print(allCont[2])
        value = re.findall('\d+', allCont[2])
        print(value[0])
