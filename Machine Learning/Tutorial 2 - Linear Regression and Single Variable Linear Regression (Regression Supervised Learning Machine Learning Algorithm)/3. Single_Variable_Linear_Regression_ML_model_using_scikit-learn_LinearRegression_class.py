import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Creating the visualisation of the dataset used for the Single Variable Linear Regression (SVLR) ML model on a graph:

# Reading the 'house_prices_with_single_independent_variable.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("house_prices_with_single_independent_variable.csv")
print(dataset)

# Plotting the (scatter) graph using the Matplotlib library
plt.xlabel('area of house/m^2')
plt.ylabel('price of house/$')
plt.scatter(dataset['area'], dataset['price'], color='red', marker='+')


# ///////////////////////////////////////////////////////////////////////////////////////


# Using the Scikit-learn Machine Learning (ML) Python library, we will now use its SVLR ML algorithm class (refer to the 
# '3. About_scikit-learn_Machine_Learning_Python_library.txt' file in the 'Tutorial 1 - What is Machine Learning' folder),
#       'LinearRegression()' from the 'sklearn.linear_model' scikit-learn ML Python library's submodule 

# to create a 'SVLR ML model' class object/instance, and allowing us to access the 
# SVLR ML algorithm class's Instance Methods such as:
#    - '.fit()'     - this Instance Method trains the SVLR ML algorithm to become a SVLR ML model
#    - '.predict()' - this Instance Method makes a prediction using the SVLR ML model
#    - '.score()'   - this Instance Method evaluates the accuracy of the SVLR ML model (see the '3. creating_a_Multiple_
#                     Variable_Linear_Regression_ML_model_to_train_with_the_One-hot_Encoded_dataset_of_nominal_categorical_
#                     variables.py' file in the 'Tutorial 4 - Machine Learning Encoding Techniques, One-hot Encoding Machine 
#                     Learning Encoding Technique and Label Encoding Machine Learning Encoding Technique (Machine Learning 
#                     Technique)' folder for more information and usage of this Instance Method)

# and the SVLR ML algorithm class's attributes such as (note that the trailing underscore ('_') is a convention 
# used in the Scikit-learn Machine Learning (ML) Python library to represent attributes):
#    - '.coef_'      - this attribute represents the value of the gradient/coefficient of the independent 
#                      variable/feature 'x', 'm', in the mathematical equation representing the SVLR ML algorithm, 
#                      'y = mx + b'
#    - '.intercept_' - this attribute represents the value of the y-intercept, 'b', in the mathematical 
#                      equation representing the SVLR ML algorithm, 
#                      'y = mx + b'


# Creating a 'SVLR ML model' class object/instance
single_variable_linear_regression_ML_model = LinearRegression()


# Training the 'SVLR ML model' class object/instance

# For the '.fit(X, y, sample_weight=None)' Instance Method of the 'SVLR ML model' class, 
# it takes 3 parameters, with the first 2 parameters are the more important ones:
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the SVLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      SVLR ML algorithm

# (Note: Why is it 'dataset[['area']]' and not 'dataset['area']'?
#        The 'X' parameter of the '.fit(X, y, sample_weight=None)' Instance Method of the 'SVLR ML model' class can 
#        only take a 2D array, and since 'dataset['area']' returns a Series, essentially a 1D array, hence it will 
#        cause an error if you use 'dataset['area']'. Only 'dataset[['area']]' will work as it returns a DataFrame, 
#        essentially a 2D array)
print(single_variable_linear_regression_ML_model.fit(dataset[['area']], dataset['price']))


# The '.coef_' attribute of the 'SVLR ML model' class shows the value of the 
# gradient/coefficient of the independent variable/feature 'x', 'm', in the mathematical equation representing the SVLR 
# ML algorithm, 'y = mx + b'      
print(single_variable_linear_regression_ML_model.coef_)                        # output: [135.78767123]


# The '.intercept_' attribute of the 'SVLR ML model' class shows the value of the 
# y-intercept, 'b', in the mathematical equation representing the Single Variable Linear 
# Regression ML algorithm, 'y = mx + b'   
print(single_variable_linear_regression_ML_model.intercept_)                   # output: 180616.43835616432


# Making predictions with the 'SVLR ML model' class object/instance

# For the '.predict(X)' Instance Method of the 'SVLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'SVLR ML model' class object/instance to make a prediction of. The output of this Instance Method is the 
#      predicted dependent variable, 'y', of the test independent variables/features by the 'SVLR ML model' class 
#      object/instance

# Hence, in this case, 
#   -> the output of this '.predict()' Instance Method of the 'SVLR ML model' class is the dependent variable, 'y' 
#      ('628715.75342466'), in the mathematical equation representing the SVLR ML algorithm, 'y = mx + b'
#   -> the input 'X' parameter of this '.predict()' Instance Method of the 'SVLR ML model' class is the independent 
#      variable/feature, 'x' ('3300') in the mathematical equation representing the SVLR ML algorithm, 'y = mx + b'

# Note: the (only) input parameter in the '.predict(X)' Instance Method of the 'SVLR ML model' class must be a 2D array. 
# So cannot be '3300', but must be '[[3300]]'
print(single_variable_linear_regression_ML_model.predict([[3300]]))            # output: [628715.75342466]


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
print(135.78767123 * 3300 + 180616.43835616432)       # = 628715.75342466   

 
# Also, as you can see, when we try to plot a (line) graph through the predicted data points of the prices and 
# the corresponding house areas on the same (scatter) graph as the dataset that we used to train the 'SVLR ML model', 
# you can see that the (line) graph through the predicted data points of the prices and the corresponding house 
# areas is actually a best fit line through the data points of the dataset that we used to train the 'SVLR ML model'.

# Since the straight line graph throough the predicted data points of the prices and the corresponding house 
# areas represents the 'SVLR ML model', this shows that the SVLR ML algorithm is essentially the best fit line (of 
# mathematical equation 'y = mx + b') through the training dataset's data points
plt.title("Matplotlib graphical visualisation of the Single\nVariable Linear Regression (SVLR) Machine Learning (ML) model")
plt.plot(dataset['area'], single_variable_linear_regression_ML_model.predict(dataset[['area']]), color='blue')

plt.savefig('matplotlib_visualisation_of_the_Single_Variable_Linear_Regression_ML_model.png', dpi=100)
plt.show()