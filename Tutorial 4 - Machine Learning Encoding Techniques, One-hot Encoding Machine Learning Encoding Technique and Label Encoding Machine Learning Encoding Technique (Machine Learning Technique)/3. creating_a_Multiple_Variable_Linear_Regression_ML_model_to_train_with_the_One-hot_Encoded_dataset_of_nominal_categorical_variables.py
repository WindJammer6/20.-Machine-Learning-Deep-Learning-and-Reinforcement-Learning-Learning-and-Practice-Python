import pandas as pd
from sklearn.linear_model import LinearRegression

# Manually creating an One-hot Encoded dataset of nominal categorical variables using dummy variables 

# Reading the 'house_prices_with_nominal_categorial_variables.csv' dataset from the CSV file using the 
# Pandas library
dataset = pd.read_csv("house_prices_with_nominal_categorial_variables.csv")
print(dataset)


# Creating the dummy variable columns, storing binary values, 0 or 1 (aka binary vectors) using the 
# '.get_dummies()' Pandas Python library function
dummy_variable_columns_storing_binary_values = pd.get_dummies(dataset['house_location'])
# dummy_variable_columns_storing_binary_values = pd.get_dummies(dataset.house_location) works too
print(dummy_variable_columns_storing_binary_values)


# Concatenating the dummy variable columns, storing binary values, 0 or 1 (aka binary vectors) with 
# the initial dataset using the '.concat()' Pandas library function
concatenated_dummy_variable_columns_storing_binary_values_with_dataset = pd.concat([dataset, dummy_variable_columns_storing_binary_values], axis='columns')
print(concatenated_dummy_variable_columns_storing_binary_values_with_dataset)


# Now we will need to drop/remove 2 columns the concatenated dummy variable columns, storing binary 
# values, 0 or 1 (aka binary vectors), and the initial dataset, which are (using the '.drop()'
# Pandas library function):
# - The 'house_location' column, which is storing the nominal categorical variables, since we no 
#   longer need it and the nominal categorical variables in the 'house_location' column cannot be 
#   fed into the ML algorithm/model 

# - Any one of the dummy variable columns ('monroe township', 'robinsville' or 'west windsor' 
#   column) (in this tutorial, we will be dropping the 'west windsor' dummy variable column), storing 
#   binary values, 0 or 1 (aka binary vectors), to prevent a common pitfall when using dummy variables, 
#   known as the Dummy Variable Trap
one_hot_encoded_dataset = concatenated_dummy_variable_columns_storing_binary_values_with_dataset.drop(['house_location', 'west windsor'], axis='columns')
print(one_hot_encoded_dataset)


# ////////////////////////////////////////////////////////////////////////////////


# Creating the Multiple Variable Linear Regression (MVLR) ML model that we will be testing our 
# One-hot Encoded dataset on (refer to the 'Tutorial 3 - Multiple Variable Linear Regression (Supervised 
# Regression Machine Learning Algorithm)' for more about the MVLR ML algorithm)

# Creating a 'MVLR ML model' class object/instance
multiple_variable_linear_regression_ML_model = LinearRegression()



# Training the 'MVLR ML model' class object/instance with the One-hot Encoded dataset

# For the '.fit(X, y, sample_weight=None)' Instance Method of the 'MVLR ML model' class, 
# it takes 3 parameters, with the first 2 parameters are the more important ones:
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MVLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MVLR ML algorithm

# Getting the independent variables/features, which is all the columns in the 'one_hot_encoded_dataset' except the 
# 'price' column (the dependent variable). Hence, here we dropped/removed the 'price' column, and taking the  
# remaining 'one_hot_encoded_dataset' as the independent variables/features

# The dependent variable is the 'price' column in the initial dataset

# (Note: What is 'one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']]'?
#        From the '3. Single_Variable_Linear_Regression_ML model_using_scikit-learn_LinearRegression_class.py' file in 
#        the 'Tutorial 2 - Linear Regression and Single Variable Linear Regression (Regression Supervised Learning 
#        Machine Learning Algorithm)' folder, we know that 'dataset[['area']]' returns a DataFrame, essentially a 2D 
#        array. Hence, 'one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']]' also returns a 2D array, 
#        but while 'dataset[['area']]' returns a 2D array DataFrame with 1 column, 'one_hot_encoded_dataset[['area', 
#        'monroe township', 'robinsville']]' returns a 2D array DataFrame with 3 columns.

#        Also, the order of the columns/independent variables/features 'area', 'monroe township', 'robinsville' in 
#        the code 'one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']]' matters, as in the mathematical 
#        equation representing the Multiple Variable Linear Regression (MVLR) ML algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... 
#        + m?*x? + b', the first column name in the code 'one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']]' 
#        will be 'x1', the second column name will be 'x2', and so on... In the mathematical equation for the MVLR ML 
#        algorithm in this context, aka the Machine Learning (ML) model for this context, it is:
#                    'y'       'x1'        'x2'      'x3' 
#                   price = m1*area + m2*monroe township + m3*robinsville + b

#        (This only applies for the MVLR ML algorithm, and not for the Single Variable Linear Regression (SVLR) ML algorithm,
#        since the SVLR ML algorithm only has 1 column/independent variable/feature):
independent_variables_or_features = one_hot_encoded_dataset.drop('price', axis='columns')
print(independent_variables_or_features)
multiple_variable_linear_regression_ML_model.fit(independent_variables_or_features, dataset['price'])
# multiple_variable_linear_regression_ML_model.fit(one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']], dataset['price']) works too



# In this tutorial, the trained MVLR ML model's coefficients and intercept/biases might look pretty 
# cryptic due to the OHE ML Encoding Technique, but it works as ML parameters 
# (i.e. coefficients/weights/biases) for the trained MVLR ML model to give accurate predictions

# The '.coef_' attribute of the 'MVLR ML model' class shows the values of the 
# gradient/coefficient of the independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1', 'm2', 'm3' ... 'm?' 
# respectively, in the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  

# (Note: The multiple coefficient values of the independent variables/features, 'm1', 'm2', 'm3' ... 'm?', will be displayed
#        in the output in order, in a Python list, for example, '[   126.89744141 -40013.97548914 -14327.56396474]', where 
#        '126.89744141' will be 'm1', '-40013.97548914' will be 'm2', and '-14327.56396474' will be 'm3')
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [   126.89744141 -40013.97548914 -14327.56396474]

# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file in the 'Tutorial 3 - Multiple Variable Linear Regression 
# (Supervised Regression Machine Learning Algorithm)' folder), 'b', in the mathematical equation representing the MVLR ML 
# algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.intercept_)                     # output: 249790.36766292527



# Making predictions with the 'MVLR ML model' class object/instance, which is trained with the One-hot Encoded dataset
# For the '.predict(X)' Instance Method of the 'MVLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'MVLR ML ML model' class object/instance to make a prediction of. The output of this Instance Method is the predicted
#      dependent variable, 'y', of the test independent variables/features by the 'MVLR ML ML model' class object/instance

# Hence, in this case, 
#   -> the output of this '.predict()' Instance Method of the 'MVLR ML model' class is the dependent variable, 'y' 
#      ('590775.63964739'), in the mathematical equation representing the MVLR ML algorithm, 
#      'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
#   -> the input 'X' parameter of this '.predict()' Instance Method of the 'MVLR ML model' class is the independent 
#      variables/features, 'x1' ('2800'), 'x2' ('0'), 'x3' ('1') in the mathematical equation representing the MVLR ML 
#      algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  

# Note: the multiple independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the '.predict(X)' 
# Instance Method of the 'MVLR ML model' class will need to be in order, in a 2D array, where '2800' will be 'x1', '0' will
# be 'x2', and '1' will be 'x3', and that they cannot be in a list/1D array like, '[2800,0,1]', but must be in a 2D 
# array like, '[[2800,0,1]]'


# In this context, 'x1' represents the 'area' independent variable/feature, 'x2' represents the 'monroe township' 
# independent variable/feature, 'x3' represents the 'robinsville' independent variable/feature

# To get a prediction of a 'price' (dependent variable) for a house in 'monroe township', the value of 'x2' (representing
# 'monroe township' independent variable/feature) must be '1', while the value of and 'x3' (representing 'robinsville' 
# independent variable/feature) must be '0'
print(multiple_variable_linear_regression_ML_model.predict([[3000,1,0]]))          # Output: 590468.71640508
# To get a prediction of a 'price' (dependent variable) for a house in 'robinsville', the value of 'x3' (representing
# 'robinsville' independent variable/feature) must be '1', while the value of and 'x2' (representing 'monroe township' 
# independent variable/feature) must be '0'
print(multiple_variable_linear_regression_ML_model.predict([[2800,0,1]]))          # Output: 590775.63964739
# To get a prediction of a 'price' (dependent variable) for a house in 'west windsor', the value of 'x2' (representing
# 'monroe township' independent variable/feature) must be '0', and the value of and 'x3' (representing 'robinsville' 
# independent variable/feature) must also be '0'
print(multiple_variable_linear_regression_ML_model.predict([[3400,0,0]]))          # Output: 681241.66845839



# Apart from the the MVLR ML algorithm class giving us access to the '.fit()' and '.predict()' Instance Methods and the
# '.coef_' and '.intercept_' attributes, the MVLR ML algorithm class also gives us access to the '.score()' Instance
# Method, which allows us to calculate how accurate the MVLR ML model is:

# How the MVLR ML algorithm class's '.score(X, y)' Instance method calculate how accurate our MVLR ML model is,
# is that it takes in 2 parameters, 
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MVLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MVLR ML algorithm

# It then uses the MVLR ML model, which is trained with the One-hot Encoded dataset, to re-predict each of the 'price'
# (dependent variable) for each of the independent variables/features in the One-hot Encoded dataset, and compare them
# with the actual 'price' (dependent variable) provided in the One-hot Encoded dataset. It then uses some mathematical
# function (not super important honestly) to calculate the percentage (%) closeness of the re-predicted 'price' (dependent 
# variable) by the MVLR ML model, which is trained with the One-hot Encoded dataset, with the actual 'price' (dependent 
# variable) provided in the One-hot Encoded dataset, as an indicator score for how accurate the MVLR ML model is 
print(multiple_variable_linear_regression_ML_model.score(independent_variables_or_features, dataset['price']))

# Output: 0.9573929037221872
# This means that the MVLR ML model is 95% accurate 


