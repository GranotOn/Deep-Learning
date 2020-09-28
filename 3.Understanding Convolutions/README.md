# Understanding Convolutions

In this lesson we will learn about the key concepts behind CNN's. This lesson is not intended to be a reference for machine learning, deep learning, convolutions or TensorFlow. The intention is to give notions to the user about these fields.

## Analogies
There are several ways to understand Convolutional Layers without using a mathematical approach. We are going to explore some of the ideas proposed by the Machine Learning community.

## Instances of Neurons
When you start to learn a programming language, one of the first phases of your development is the learning and application of functions. Instead of rewriting pieces of code everytime that you would, a good student is encouraged to code using functional programming, keeping the code organized, clear and concise. CNNs can be thought of as a simplification of what is really going on, a special kind of neural network which uses identical copies of the same neuron. These copies include the same parameters (shared weights and biases) and activation functions.

## Location and type of connections
In a fully connected layer NN, each neuron in the current layer is connected to every neuron in the previous layer, and each connection has it's own weight. This is a general purpose connection pattern and makes no assumptions about the features in the input data thus not taking any advantage that the knowledge of the data being used can bring. These types of layers are also very expensive in terms of memory and computation.

In contrast, in a convolutional layer each neuron is only connected to a few nearby local neurons in the previous layer, and the same set of weights is used to connect to them. For example, in the following image, the neurons in the h1 layer are connected only to some input units (pixels).

![convolutional_diagram](https://ibm.box.com/shared/static/mev168hepixnmc9zhh4hsr3t2ks3rpcc.png)