# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


Matrix_one = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
Matrix_two = tf.constant([[2, 2, 2], [2, 2, 2], [2, 2, 2]])

# Addition


@tf.function
def add():
    add_1_operation = tf.add(Matrix_one, Matrix_two)
    return add_1_operation


print("Defined using tensorflow function:\n %s \n " % add())
print("Defined using normal expressions: \n %s \n" % (Matrix_one + Matrix_two))


# Subtraction

@tf.function
def subtract():
    subtract_1_operation = tf.subtract(Matrix_one, Matrix_two)
    return subtract_1_operation


print("Subtracting A from B: \n %s \n" % subtract())

# Multiplication


@tf.function
def multiply():
    multiply_1_operation = tf.multiply(Matrix_one, Matrix_two)
    return multiply_1_operation


@tf.function
def mathmul():
    return tf.matmul(Matrix_one, Matrix_two)


print("Multiplication using tf.multiply: \n %s \n" %
      multiply())  # Hadamard product
print("Multiplication using tf.matmul: \n %s \n" %
      mathmul())  # Matrices products
