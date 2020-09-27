# Linear Regression with TensorFlow

Linear regression, in layman terms, is the approximation of a linear model used to describe the relationship between two or more variables. In it's simplest form there are two variables: 
- the dependent variable, which can be seen as the "state" or "final goal" that we study and try to predict.

- the independent variables, also known as explanatory variables, which can be seen as the "causes" of the "states"

When more than one independent variable is present, the process is called multiple linear regression.
When multiple dependent variables are predicted, the process is known as mutivariate linear regression.

The equation of a simple linear model is  
![equation](http://www.sciweavers.org/upload/Tex2Img_1601203069/render.png)    

- y is the dependent variable
- x is the independent variable
- a, b are paremeters we adjust (a is slope, b is intercept)  

You can interpret this equation as y being a function of x, or y being dependent on x. 

In the first file we will plot this equation.

Linear relations were used to try describe and quantify many observable physical phenomena.

In the second file we will see a simple example using a sample dataset.

## Dependencies

Packages:

```
pip install matplotlib
pip install pandas
pip install pylab
pip install numpy
pip install tensorflow=2.2.0
```

Dataset:
```
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
```

## File Structure (in order)

- 0_plot.py