# -*- coding=utf-8 -*-'''
import matplotlib.pyplot as plt
import re

from urllib3.connectionpool import xrange

logs = open('loss').read()
# print logs

pattern = re.compile('\*?(\d+) (.*?) (.*?) (.*?) (.*?)\n', re.S)
result = re.findall(pattern, logs)
print(len(result))
# end to end
img = {}
c = 0
iteration = []
loss = []

iteration = []
loss = []
for i in range(0, 512):
    iteration.append(result[i][0])
    loss.append(result[i][2])

colors = 'navy'
plt.clf()
plt.plot(iteration, loss, color=colors)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.xlim([0.0, 611500.0])
plt.ylim([0.0, 1.5])
plt.title('End to end')
plt.legend(loc="lower left")
plt.show()
