# Logistic Regression with TensorFlow

- **Recall Linear Regression with TensorFlow before starting**

## What is different between Linear and Logistic Regression?
While Linear Regression is suited for estimating continuous values (e.g estimating house price), it is not the best tool for predicting the class in which an observed data point belongs.  
In order to provide estimate for classification, we need some sort of guidance on what would be the most probable class for that data point. For this, we use Logistic Regression.    

Logistic Regression is a variation on Linear Regression, and is useful when the observed dependent variable (y) is categorical.  
It produces a formula that predicts the probability of the class label as a function of the independent variables.  

Despite the name logistic regression, it is actually a probabilistic classification model. Logistic regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into a probability with the following function:

- <img src="https://latex.codecogs.com/gif.latex?P(s | O_t )=\text { Probability of a sensor reading value when sleep onset is observed at a time bin } t " />
