# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

input = tf.Variable(tf.random.normal([1, 10, 10, 1])) #image
filter = tf.Variable(tf.random.normal([3, 3, 1, 1]))
valid = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
same = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")

print('Input \n {0} \n'.format(input.numpy()))
print('Filter \n {0} \n'.format(filter.numpy()))
print("Result/Feature Map with valid positions \n")
print(valid.numpy()) # 8x8
print('\n')
print("Result/Feature Map with padding \n")
print(same.numpy()) # 10x10