# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

'''
Operations are nodes that represent the mathematical
operations over the tensors on a graph.
'''

a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)


print ('c =: %s' % c)
'''
c =: tf.Tensor([7], shape=(1,), dtype=int32)
'''

print ('d =: %s' % d)
'''
d =: tf.Tensor([3], shape=(1,), dtype=int32)
'''