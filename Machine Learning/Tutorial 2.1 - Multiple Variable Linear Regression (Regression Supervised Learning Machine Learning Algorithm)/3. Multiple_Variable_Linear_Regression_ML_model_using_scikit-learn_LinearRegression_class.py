import pandas as pd
import math
from sklearn.linear_model import LinearRegression

# Cleaning and preparing the 'house_prices_with_multiple_independent_variables.csv' dataset:

# Reading the 'house_prices_with_multiple_independent_variables.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv('house_prices_with_multiple_independent_variables.csv')

# As we can see, the 'house_prices_with_multiple_independent_variables.csv' dataset is messy and has a missing NaN/NA/Null 
# value/cell in the 'bedrooms' column in the dataset. Hence, we will need to clean and prepare this dataset, by 
# handling this missing NaN/NA/Null value/cell in the 'bedrooms' column in the dataset.
        #    area  bedrooms  age   price
        # 0  2600       3.0   20  550000
        # 1  3000       4.0   15  565000
        # 2  3200       NaN   18  610000
        # 3  3600       3.0   30  595000
        # 4  4000       5.0    8  760000
print(dataset)

# It is decided that we will use the median (using the '.median()' Pandas function) of all the values in the 
# 'bedrooms' column in the dataset as a 'fill in' of this missing NaN/NA/Null value/cell in the 'bedrooms' 
# column in the dataset, to handle the missing NaN/NA/Null value/cell in the 'bedrooms' column in the dataset.
median_bedrooms = dataset['bedrooms'].median()
print(median_bedrooms)                                          # output: 3.5

# However, it does not make sense for the number of 'bedrooms''s value to be '3.5'. Hence, it is decided that
# we will round down the value of the median of all the values in the 'bedrooms' column in the dataset that we
# will be using as a 'fill in' of this missing NaN/NA/Null value/cell in the 'bedrooms' column in the dataset.
always_rounded_down_median_bedrooms = math.floor(median_bedrooms)
print(always_rounded_down_median_bedrooms)                      # output: 3

# Using the '.fillna()' Pandas function to 'fill in' any missing NaN/NA/Null value/cell in a column, and in this
# case, the missing NaN/NA/Null value/cell in the 'bedrooms' column in the dataset with the always rounded down
# median value stored in the 'always_rounded_down_median_bedrooms' variable
dataset['bedrooms'] = dataset['bedrooms'].fillna(always_rounded_down_median_bedrooms)

# Here is the cleaned and prepared dataset, with the missing NaN/NA/Null value/cell in the 'bedrooms' column in 
# the dataset now handled, and is ready to be used to train the Multiple Variable Linear Regression (MVLR) ML 
# algorithm to become a MVLR ML model.
print(dataset)


# ////////////////////////////////////////////////////////////////////////////////////////////////


# Using the Scikit-learn Machine Learning (ML) Python library, we will now use its Multiple Variable Linear 
# Regression (MVLR) ML algorithm class (refer to the '3. About_scikit-learn_Machine_Learning_Python_library.txt' 
# file in the 'Tutorial 1 - What is Machine Learning' folder),
#       'LinearRegression()' from the 'sklearn.linear_model' scikit-learn ML Python library's submodule 

# to create a 'MVLR ML model' class object/instance, and allowing us to access the 
# MVLR ML algorithm class's Instance Methods such as:
#    - '.fit()'     - this Instance Method trains the MVLR algorithm to become a MVLR ML model
#    - '.predict()' - this Instance Method makes a prediction using the MVLR ML model
#    - '.score()'   - this Instance Method evaluates the accuracy of the MVLR ML model (see the '3. creating_a_Multiple_
#                     Variable_Linear_Regression_ML_model_to_train_with_the_One-hot_Encoded_dataset_of_nominal_categorical_
#                     variables.py' file in the 'Tutorial 4 - Machine Learning Encoding Techniques, One-hot Encoding Machine 
#                     Learning Encoding Technique and Label Encoding Machine Learning Encoding Technique (Machine Learning 
#                     Technique)' folder for more information and usage of this Instance Method)

# and the MVLR ML algorithm class's attributes such as (note that the trailing underscore ('_') is a convention used 
# in the Scikit-learn Machine Learning (ML) Python library to represent attributes):
#    - '.coef_'      - this attribute represents the values of the gradients/coefficients of the independent 
#                      variables/features 'x1', 'x2', 'x3' ... 'x? , 'm1', 'm2', 'm3' ... 'm?' respectively, in 
#                      the mathematical equation representing the MVLR ML algorithm, 
#                      'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
#    - '.intercept_' - this attribute represents the value of the intercept (NOT y-intercept, refer to the 
#                      '1. What_is_Multiple_Variable_Linear_Regression.txt' file), 'b', in the mathematical equation 
#                      representing the MVLR ML algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  


# Creating a 'MVLR ML model' class object/instance
multiple_variable_linear_regression_ML_model = LinearRegression()


# Training the 'MVLR ML model' class object/instance

# For the '.fit(X, y, sample_weight=None)' Instance Method of the 'MVLR ML model' class, 
# it takes 3 parameters, with the first 2 parameters are the more important ones:
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MVLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MVLR ML algorithm

# (Note: What is 'dataset[['area', 'bedrooms', 'age']]'?
#        From the '3. Single_Variable_Linear_Regression_ML model_using_scikit-learn_LinearRegression_class.py' file 
#        in the 'Tutorial 2 - Linear Regression and Single Variable Linear Regression (Regression Supervised 
#        Learning Machine Learning Algorithm)' folder, we know that 'dataset[['area']]' returns a DataFrame, 
#        essentially a 2D array. Hence, 'dataset[['area', 'bedrooms', 'age']]' also returns a 2D array, but while 
#        'dataset[['area']]' returns a 2D array DataFrame with 1 column, 'dataset[['area', 'bedrooms', 'age']]' 
#        returns a 2D array DataFrame with 3 columns.

#        Also, the order of the columns/independent variables/features 'area', 'bedrooms', and 'age' in the code 
#        'dataset[['area', 'bedrooms', 'age']]' matters, as in the mathematical equation representing the Multiple Variable 
#        Linear Regression (MVLR) ML algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b', the first column name in the code 
#        'dataset[['area', 'bedrooms', 'age']]' will be 'x1', the second column name will be 'x2', and so on... In the
#        mathematical equation for the MVLR ML algorithm in this context, aka the Machine Learning (ML) ML model for this 
#        context, it is:
#                    'y'       'x1'        'x2'      'x3' 
#                   price = m1*area + m2*bedroom + m3*age + b

#        (This only applies for the MVLR ML algorithm, and not for the Single Variable Linear Regression (SVLR) ML algorithm,
#        since the SVLR ML algorithm only has 1 column/independent variable/feature):
multiple_variable_linear_regression_ML_model.fit(dataset[['area', 'bedrooms', 'age']], dataset['price'])
print(multiple_variable_linear_regression_ML_model)


# The '.coef_' attribute of the 'MVLR ML model' class shows the values of the 
# gradient/coefficient of the independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1', 'm2', 'm3' ... 'm?' respectively, 
# in the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  

# (Note: The multiple coefficient values of the independent variables/features, 'm1', 'm2', 'm3' ... 'm?', will be displayed
#        in the output in order, in a Python list, for example, '[   137.25 -26025.    -6825.  ]', where '137.25' will be 
#        'm1', '-26025' will be 'm2', and '-6825' will be 'm3')
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [   137.25 -26025.    -6825.  ]


# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file), 'b', in the mathematical equation representing the MVLR ML 
# algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.intercept_)                     # output: 383724.9999999998


# Making predictions with the 'MVLR ML model' class object/instance

# For the '.predict(X)' Instance Method of the 'MVLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'MVLR ML model' class object/instance to make a prediction of. The output of this Instance Method is the predicted
#      dependent variable, 'y', of the test independent variables/features by the 'MVLR ML model' class object/instance

# Hence, in this case, 
#   -> the output of this '.predict()' Instance Method of the 'MVLR ML model' class is the dependent variable, 'y' 
#      ('444400.'), in the mathematical equation representing the MVLR ML algorithm, 
#      'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
#   -> the input 'X' parameter of this '.predict()' Instance Method of the 'MVLR ML model' class is the independent 
#      variables/features, 'x1' ('3000'), 'x2' ('3'), 'x3' ('40') in the mathematical equation representing the MVLR ML 
#      algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  

# Note: the multiple independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the '.predict(X)' 
# Instance Method of the 'MVLR ML model' class will need to be in order, in a 2D array, where '3000' will be 'x1', '3' will
# be 'x2', and '40' will be 'x3', and that they cannot be in a list/1D array like, '[3000, 3, 40]', but must be in a 2D 
# array like, '[[3000, 3, 40]]'
print(multiple_variable_linear_regression_ML_model.predict([[3000, 3, 40]]))       # output: [444400.]


# /////////////////////////////////////////////////////////////////////////////////////////////////


# Some key observations from all this:

# So how is this predicted value/output made by the 'Multiple Variable Linear Regression (MVLR) ML model' class 
# object/instance?
# Essentially, looking at the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  , since:
# - the '.coef_' attribute of the 'MVLR ML model' class shows the value of the gradients/coefficients of the independent 
#   variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1' ('137.25'), 'm2' ('-26025'), 'm3' ('-6825.') respectively 
# - the input 'X' parameter of this '.predict()' Instance Method of the 'MVLR ML model' class is the independent 
#   variables/features, 'x1' ('3000'), 'x2' ('3'), 'x3' ('40')
# - the '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
#   '1. What_is_Multiple_Variable_Linear_Regression.txt' file), 'b' ('383724.9999999998')

# And since the output of this '.predict()' Instance Method of the 'MVLR ML model' class is the dependent variable, 'y' 
# ('444400.'), to prove that 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b' is the mathematical equation representing the 
# MVLR ML algorithm, we can substitude the values of of 'm1', 'm2', 'm3', 'x1', 'x2', 'x3' and 'b' into the 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b' mathematical equation to find the same value of 'y', which is also
# the predicted value/output made by the 'MVLR ML model' class object/instance shown via the '.predict()' Instance Method of 
# the 'MVLR ML model' class

#      'm1'  * 'x1' +  'm2'  *  'x2' +  'm3'  * 'x3' +       'b'                 =   y
print(137.25 * 3000 + -26025 *   3   + -6825. *  40  + 383724.9999999998)      # = 444400.   

