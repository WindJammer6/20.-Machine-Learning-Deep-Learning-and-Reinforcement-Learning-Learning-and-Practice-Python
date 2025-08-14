# Question 1:
# Given Canada's net national income per capita from the year 1970 to 2016, in the dataset 
# 'canada_per_capita_income.csv', predict Canada's net national income per capita in the year 2020 using 
# the Single Variable Linear Regression (SVLR) Supervised learning Regression Machine Learning (ML) algorithm.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Creating the visualisation of the dataset used for the Single Variable Linear Regression (SVLR) ML model on a graph:

# Reading the 'canada_per_capita_income.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("canada_per_capita_income.csv")
print(dataset)

# Plotting the (scatter) graph using the Matplotlib library
plt.xlabel('year')
plt.ylabel('per capita income/US$')
plt.scatter(dataset['year'], dataset['per capita income (US$)'], color='red', marker='+')


# ///////////////////////////////////////////////////////////////////////////////////////

# Creating a 'SVLR ML model' class object/instance
single_variable_linear_regression_ML_model = LinearRegression()


# Training the 'SVLR ML model' class object/instance
print(single_variable_linear_regression_ML_model.fit(dataset[['year']], dataset['per capita income (US$)']))


# The '.coef_' attribute of the 'SVLR ML model' class shows the value of the gradient/coefficient of 'x', 'm', in 
# the mathematical equation representing the SVLR ML algorithm, 'y = mx + b'      
print(single_variable_linear_regression_ML_model.coef_)                # output: [828.46507522]


# The '.intercept_' attribute of the 'SVLR ML model' class shows the value of the y-intercept, 'b', in the 
# mathematical equation representing the SVLR ML algorithm, 'y = mx + b'   
print(single_variable_linear_regression_ML_model.intercept_)           # output: -1632210.7578554575


# Making predictions with the 'SVLR ML model' class object/instance, predicting Canada's net national income 
# per capita in the year 2020 using the SVLR Supervised learning Regression Machine Learning (ML) algorithm:
print(single_variable_linear_regression_ML_model.predict([[2020]]))    #output: [41288.69409442]


# ////////////////////////////////////////////////////////////////////////////////////////


# Some key observations from all this:

# So how is this predicted value/output made by the 'Single Variable Linear Regression (SVLR) ML model' class 
# object/instance?
# Essentially, looking at the mathematical equation representing the SVLR ML algorithm, 'y = mx + b', since:
# - the '.coef_' attribute of the 'SVLR ML model' class shows the value of the gradient/coefficient of the independent
#   variable/feature 'x', 'm' ('135.78767123')
# - the input 'X' parameter of this '.predict()' Instance Method of the 'SVLR ML model' class is the independent 
#   variable/feature, 'x' ('3300')
# - the '.intercept_' attribute of the 'SVLR ML model' class shows the value of the y-intercept, 'b' ('180616.43835616432')

# And since the output of this '.predict()' Instance Method of the 'SVLR ML model' class is the dependent variable, 
# 'y' ('628715.75342466'), to prove that 'y = mx + b' is the mathematical equation representing the SVLR ML algorithm, 
# we can substitude the values of 'm', 'x', and 'b' into the 'y = mx + b' mathematical equation to find the same 
# value of 'y', which is also the predicted value/output made by the 'SVLR ML model' class object/instance shown
# via the '.predict()' Instance Method of the 'SVLR ML model' class

#        'm'       *  'x' +         'b'                 =       y
print(828.46507522 * 2020 + -1632210.7578554575)       # = 41288.69409442

 
# Also, as you can see, when we try to plot a (line) graph through the predicted data points of the prices and 
# the corresponding house areas on the same (scatter) graph as the dataset that we used to train the 'SVLR ML model', 
# you can see that the (line) graph through the predicted data points of the prices and the corresponding house 
# areas is actually a best fit line through the data points of the dataset that we used to train the 'SVLR ML model'.

# Since the straight line graph through the predicted data points of the prices and the corresponding house 
# areas represents the 'SVLR ML model', this shows that the SVLR ML algorithm is essentially the best fit line (of 
# mathematical equation 'y = mx + b') through the training dataset's data points
plt.title("Matplotlib graphical visualisation of the Single\nVariable Linear Regression (SVLR) Machine Learning (ML) model\nfor Excercise 1")
plt.plot(dataset['year'], single_variable_linear_regression_ML_model.predict(dataset[['year']]), color='blue')

plt.savefig('matplotlib_visualisation_of_the_Single_Variable_Linear_Regression_ML_model_for_excercise_1.png', dpi=100)
plt.show()