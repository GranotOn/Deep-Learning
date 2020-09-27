# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# Defining multidimensional arrays using TensorFlow

Scalar = tf.constant(2)
Vector = tf.constant([5, 6, 2])
Matrix = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
Tensor = tf.constant([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[4, 5, 6], [
                     5, 6, 7], [6, 7, 8]], [[7, 8, 9], [8, 9, 10], [9, 10, 11]]])

# tf.Tensor(2, shape=(), dtype=int32)
print("Scalar (1 entry): \n %s \n" % Scalar)

# tf.Tensor([5 6 2], shape=(3,), dtype=int32)
print("Vector (3 entries): \n %s \n" % Vector)

print("Matrix (3x3 entries):\n %s \n" % Matrix)
'''
 tf.Tensor(
[[1 2 3]
 [2 3 4]
 [3 4 5]], shape=(3, 3), dtype=int32)
'''
print("Tensor (3x3x3 entries) :\n %s \n" % Tensor)
'''
tf.Tensor(
[[[ 1  2  3]
  [ 2  3  4]
  [ 3  4  5]]

 [[ 4  5  6]
  [ 5  6  7]
  [ 6  7  8]]

 [[ 7  8  9]
  [ 8  9 10]
  [ 9 10 11]]], shape=(3, 3, 3), dtype=int32)
'''

# tf.shape returns the shape of our data structure

print(Tensor.shape)  # (3,3,3)
