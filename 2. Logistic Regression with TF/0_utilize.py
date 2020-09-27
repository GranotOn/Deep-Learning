# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time
import numpy as np
import pandas as pd
import tensorflow as tf



if not tf.__version__ == '2.2.0':
    print(tf.__version__)
    raise ValueError('please upgrade to TensorFlow 2.2.0')

'''
We will load the 'iris' dataset (built-in).
We will then seperate the dataset into xs and ys, train and test them (pseudo) randomly.
Check README for in-depth explanation about iris
'''

iris = load_iris()
# Python Slicing notation; data is all independent variables; target (0-2) is species (dependent)
iris_X, iris_Y = iris.data[:-1, :], iris.target[:-1]

# Pandas.get_dummies(data) converts categorical variable into dummy variables.
# With addition to .values it basically creates a matrix
iris_Y = pd.get_dummies(iris_Y).values

# sklearn test_train_split() splits arrays or matrices into random train and test subsets
trainX, testX, trainY, testY = train_test_split(
    iris_X, iris_Y, test_size=0.33, random_state=42)

'''
Now we define x and y. These variables will hold our iris data (both the features and the label matrices).
We also need to give them shapes which correspond to the shape of our data.
'''

# numFeatures is the number of features in our input data.
# In the iris dataset, this number is '4'.

numFeatures = trainX.shape[1]
print('numFeaturs: ', numFeatures)  # '4'

# numLabels is the number of classes our data points can be in
# In the iris dataset, this number is '3'.

numLabels = trainY.shape[1]
print('numLabels ', numLabels)  # '3'

# Numpy identity(n) creates an identity matrix * n
# Notice the diffrences in initializing X and yGold.
# Iris has 4 features, so X is a tensor to hold our data
X = tf.Variable(np.identity(numFeatures),
                tf.TensorShape(numFeatures), dtype='float32')
yGold = tf.Variable(np.array([1, 1, 1]), shape=tf.TensorShape(
    numLabels), dtype='float32')  # This will be our correct answers matrix for 3 classes

'''
Like Linear Regression, we need a shared variable weight matrix for Logistic Regression.
We initialize both W and b as tensors full of zeros. Since we are going to learn W and b, their initial value
does not matter too much (like a,b when we did Linear Regression).

These variables are the objects which define the structure of our regression model,
and we can save them after they have been trained so we can reuse them later.

TLDR; We define two TF variables as our parameters, which will hold the weights and biases of our 
Logistic regression and continually update.
'''

# W has shape [4,3] because we want to multiply the 4-dimensional input vectors by it
# to produce 3 dimensional vectors.
# b has shape of [3] so we cann later add it to output.
W = tf.Variable(tf.zeros([4, 3]))  # 4-dimensional input and 3 classes
b = tf.Variable(tf.zeros([3]))  # 3-dimensional output [0,0,1],[0,1,0][1,0,0]

# Randomly sample from a normal distribution with standard deviation .01
weights = tf.Variable(tf.random.normal(
    [numFeatures, numLabels], mean=0, stddev=0.01, name="weights"), dtype='float32')

bias = tf.Variable(tf.random.normal([1,numLabels],
                                    mean=0.,
                                    stddev=0.01,
                                    name="bias"))

'''
We now define our operations in order to properly run the Logistic Regression.
Logistic Regression is typically thought of as sigmoid function over linear regression.

For the sake of clarity we can have it broken to three main components:
1) A weight times features matrix multiplication operation.
2) A summation of the weighted features and a bias term.
3) The application of a sigmoid function
'''

def logistic_regression(x):
    apply_weights_OP = tf.matmul(X, weights, name="apply_weights") # matmul = matrice multiplication
    add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation") # sigmoid function
    return activation_OP

'''
The learning algorithm is how we search for the best weight vector (W).
This search is an optimization problem looking for the hypothesis that optimizes an error/cost measure.
We want to minimize the Cost/Loss.
'''
# Number of Epochs (time unit)
numEpochs = 700

# Defining our learning rate iterations (decay)
learningRate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0008,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

# Cost function (Remember: the loss function for Linear Regression)

loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
optimizer = tf.keras.optimizers.SGD(learningRate)

# Let's add additional operations to keep track of our model's efficiency over time.

#Accuray metric
def accuracy(y_pred, y_true):
    #Predicted class is the index of the highest score in the prediction vector (i.e. argmax).
    print('y_pred : ', y_pred)
    print('y_true : ', y_true)
    correct_prediction = tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_true, -1))

    # reduce_mean() = compute the mean of elements across dimensions of tensor
    # inside we cast the correct prediction into a float32
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# We first wrap computation inside a GradientTape for automatic differntation.
# Then we compute gradients and update W and b

# Optimization process
def run_optimization(x, y):
    with tf.GradientTape() as tape:
        pred = logistic_regression(x)
        loss = loss_object(pred, y)
    
    gradients = tape.gradient(loss, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))

'''
Now we can move to actually running our operations.
We will start with the operations involved in the prediction phase.
'''


# Initialize reporting variables
display_step = 10
epoch_values = []
accuracy_values = []
loss_values = []
loss = 0
diff = 1

#Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .000001:
        print('change in loss %g; convergence.' %diff)
        break
    else:
        # Run training step
        run_optimization(X, yGold)

        #Report occasional stats
        if i % display_step == 0:
            # Add epoch to epoch values
            epoch_values.append(i)
            
            pred = logistic_regression(X)
            
            newLoss = loss_object(pred, yGold)
            
            #Add Loss to live graphing variable
            loss_values.append(newLoss)

            #Generate accuracy stats on test data
            acc = accuracy(pred, yGold)
            accuracy_values.append(acc)

            #Re-assign values for variables
            diff = abs(newLoss - loss)
            loss = newLoss

            #Generate oprint statements
            print('step %d, training accuracy %g, loss %g, change in loss %g'%(i,acc,newLoss,diff))
        

# How well do we perform on held-out test data?
print('Final accuracy on test set: %s' %str(acc))

plt.plot([np.mean(loss_values[i-50:i]) for i in range (len(loss_values))])
plt.show()
