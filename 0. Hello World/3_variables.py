# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Creating a simple counter

v = tf.Variable(0)


'''
increment_by_one is a python method which will
internally call tf.add, to increment our counter.
'''

@tf.function
def increment_by_one(v):
    v = tf.add(v, 1)
    return v

# Increment our counter 3 times using increment_by_one

for i in range(3):
    v = increment_by_one(v)
    print(v)

'''
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
'''


