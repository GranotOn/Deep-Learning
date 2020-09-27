# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Import dataset using Pandas
df = pd.read_csv("FuelConsumption.csv")

# Check integrity: print(df.head())

'''
Assume we want to use linear regression in-order to predict
CO2Emission of cars based on their engine size.
Let's define X and Y for the linear regression:
'''

# numpy 'assanarray()' method converts input into ndarray
# ndarray is an array object, represents a multi-dimensional array.
train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])


# Initialize 'a' and 'b' with random guess
a = tf.Variable(20.0)
b = tf.Variable(30.2)

# Our linear equation
def h(x):
    y = a*x + b
    return y

'''
We need a loss function for our regression, so we can
train our model to better fit our data.

In a linear regression, we minimize the squared error of
the difference between the predicted values (obtained from the equation)
and the target values (the data that we have).
In other words, we want to minimize the squared of the predicted
values minus the target value. So we define the equation to be
minimized as loss.
'''

# To find value of our loss, we use tf.reduce_mean()
# tf.reduce_mean() finds the mean of a multidimensional tensor.

def loss_object(y, train_y):
    return tf.reduce_mean(tf.square(y - train_y))

    # Below is a predifined method offered by TF to calculte loss function
    # loss_object = tf.keras.losses.MeanSquaredLogrithmicError()

'''
Below we start training and run the graph.
We use GradientTape to calculate gradients
'''

learning_rate = 0.01
train_data = []
loss_values = []

# Steps of looping through all your data to update the parameters
training_epochs = 200

# train model
for epoch in range(training_epochs):
    # GradientTape records operations for automatic differntiations
    with tf.GradientTape() as tape:
        # predict y using a, b as guesses
        y_predicted = h(train_x)

        # determine loss object from actual value
        loss_value = loss_object(train_y, y_predicted)

        # append the loss to the loss list
        loss_values.append(loss_value)

        # get gradients
        gradients = tape.gradient(loss_value, [b, a])
        
        # compute and adjust weights
        b.assign_sub(gradients[0] * learning_rate)
        a.assign_sub(gradients[1] * learning_rate)

        if (epoch % 5 == 0):
            train_data.append([a, b])

# Plot the loss values to see changes over time.
plt.plot(loss_values, 'ro')

# Visualize how to coeffiecient and interecept have changed

cr, cg, cb = (1.0, 1.0, 1.0)

for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)

    if (cb > 1.0): cb = 1.0
    if (cg < 0.0): cg = 0.0

    # sequence unpacking
    [a, b] = f

    f_y = np.vectorize(lambda x: a*x + b)(train_x)

    line = plt.plot(train_x, f_y)

    plt.setp(line, color=(cr, cg, cb))

plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()



