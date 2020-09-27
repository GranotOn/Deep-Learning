# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


if not tf.__version__ == '2.2.0':
    print(tf.__version__)
    raise ValueError(
        'please upgrade to TensorFlow 2.2.0, or restart your Kernel (Kernel->Restart & Clear Output)')

# Adding 2 constants to our graph

a = tf.constant([2], name='constant_a')
b = tf.constant([3], name='constant_b')

# Examining tensor a

print(a)  # Should print name, shape and type of the tensor in the graph.
# Should print 2, as in the value of the tensor in the graph.
tf.print(a.numpy()[0])

'''
Annotating the python functions with @tf.function creates
a TensorFlow static execution graph for the function.
Which means TensorFlow will transform the function add into
TensorFlow control flow, which then defines the TensorFlow 
static exceution graph.
'''

@tf.function
def add(a, b):
    c = tf.add(a, b)
    # c = a + b is also a way to define the sum of the terms.
    print(c) # Should print operation, shape and type.
    return c

result = add(a, b)
tf.print(result[0]) # Should print 5, as in 2 + 3