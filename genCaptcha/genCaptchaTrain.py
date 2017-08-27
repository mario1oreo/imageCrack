# coding=utf-8
import os
import random

import datetime
from filecmp import cmp

import tensorflow as tf
import numpy as np
from PIL import Image

# 图片的高度
IMAGE_HEIGHT = 60
# 图片的宽度
IMAGE_WIDTH = 160
# 图片的验证码字符长度
MAX_CAPTCHA = 4
# 验证码的字符集长度
CHAR_SET_LEN = 75

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


# 随机获取文件名和图片
def get_name_and_image():
    fileDir = 'D:/work/captcha/genCaptcha/originalImageTemp/'
    random_file = random.randint(0, 49056)
    # fileDir = 'd:/work/captcha/genCaptcha/test/'
    # random_file = random.randint(0, 29650)
    all_image = os.listdir(fileDir)
    base = os.path.basename(fileDir + all_image[random_file])
    name = os.path.splitext(base)[0]
    image = Image.open(fileDir + all_image[random_file])
    image = np.array(image)
    # print('image:'+image)
    # print('name:' + name)
    return name, image


# 名字转成向量
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * 75 + ord(c) - 48
        vector[idx] = 1
    return vector


# 向量转成名字
def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 48)
        name.append(a)
    return "".join(name)


# 生成一个训练的batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1 * (image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # 3 convert layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        baseAcc = 0.8
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            print(step, loss_)

            # 每100 Step 计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(400)
                acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
                print(step, acc, datetime.datetime.now())

                if acc > 0.9999:
                    saver.save(sess, "./crack_capcha-step-" + str(step) + "-" + str(acc) + ".model", global_step=step)
                    break
                elif acc > baseAcc:
                    baseAcc = acc
                    saver.save(sess, "./crack_capcha-step-" + str(step) + "-" + str(acc) + ".model", global_step=step)

            step += 1


train_crack_captcha_cnn()

# 训练完成后#掉train_crack_captcha_cnn()，取消下面的注释，开始预测，注意更改预测集目录

# def crack_captcha():
#     output = crack_captcha_cnn()
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint('.'))
#         n = 0
#         trueNum = 0
#         while n <= 450:
#             text, image = get_name_and_image()
#             image = 1 * (image.flatten())
#             predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
#             text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
#             vec = text_list[0].tolist()
#             predict_text = vec2name(vec)
#             print("正确: {}  预测: {}  结果: {}".format(text, predict_text,text == predict_text))
#             if (text == predict_text):
#                 trueNum += 1
#             n += 1
#         print("准确率：" + str(trueNum/n))
#
# crack_captcha()
