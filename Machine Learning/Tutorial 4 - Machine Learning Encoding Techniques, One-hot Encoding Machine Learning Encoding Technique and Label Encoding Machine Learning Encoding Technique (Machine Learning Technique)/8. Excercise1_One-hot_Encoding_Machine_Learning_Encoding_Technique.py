# Question 1:
# In the 'car_prices_with_nominal_categorical_variables.csv' file, there are prices for 3 different car models.
 
# Plot the data points of the 'Mileage' column and 'sell_price' column on a scatter plot to see if the
# Linear Regression Machine Learning (ML) algorithm can be applied for this dataset (This step is crucial, as
# in Machine Learning, choosing the correct ML algorithm for a particular dataset/problem is an important skill).
 
# If yes, explain why, and build a Linear Regression ML model that can answer following questions,
# 1) Predict sell price of a Mercedez Benz C class that is, 4 years old with mileage 45000
# 2) Predict sell price of a BMW X5 that is, 7 years old with mileage 86000
# 3) Tell me the score (accuracy) of your Linear Regression ML model. (Hint: use the '.score()' Instance Method
#    of the 'LinearRegression' class)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

# Reading the 'car_prices_with_nominal_categorical_variables.csv' dataset from the CSV file using the Pandas 
# library
dataset = pd.read_csv("car_prices_with_nominal_categorical_variables.csv")
print(dataset)


# ///////////////////////////////////////////////////////////////////////////////////////


# Plotting the data points of the 'mileage' column and 'sell_price' column on a scatter plot
x = dataset['mileage']
y = dataset['sell_price']

plt.scatter(x,y)

plt.title('Scatter graph of Mileage against Sell Price of Cars for Excercise 1')
plt.xlabel('Mileage')
plt.ylabel('Sell Price')

plt.savefig('scatter_graph_of_mileage_against_sell_price_of_cars_for_excercise_1.png', dpi=100)
plt.show()

# Yes, Linear Regression Machine Learning (ML) algorithm can be used here, as there is a clear linear relationship
# between the 'mileage' independent variable and 'sell_price' dependent variable of the cars provided in the 
# 'car_prices_with_nominal_categorical_variables.csv' dataset as seen from the scatter plot.


# ///////////////////////////////////////////////////////////////////////////////////////


# Not very important, just due to Scikit-learn documentation, in order to apply the Scikit-learn's 'OneHotEncoder' 
# class, there is a need to apply the Scikit-learn 'ColumnTransformer' class beforehand whcih helps to specify 
# which column in a dataset to apply the Scikit-learn's 'OneHotEncoder' class to 
columntransformer_with_onehotencoder = ColumnTransformer(transformers=[("car_model", OneHotEncoder(), [0])], remainder='passthrough')



# Using the '.fit_transform()' Instance method of the Scikit-learn's 'OneHotEncoder' class to 'fit' the nominal
# categorical variables, in the 'car_model' column, into the 'OneHotEncoder' to do One-hot Encoding on the 
# nominal categorical variables. The Scikit-learn's 'OneHotEncoder' class 'fits' the nominal categorical 
# variables in ASCII order.

# The 'fitted' nominal categorical variables, in the 'car_model' column, into the 'OneHotEncoder' is then
# assigned back into the initial dataset
concatenated_dummy_variable_columns_storing_binary_values_with_dataset = columntransformer_with_onehotencoder.fit_transform(dataset)
print(concatenated_dummy_variable_columns_storing_binary_values_with_dataset)

# Output: 
# [[0.00e+00 1.00e+00 0.00e+00 6.90e+04 1.80e+04 6.00e+00]
#  [0.00e+00 1.00e+00 0.00e+00 3.50e+04 3.40e+04 3.00e+00]
#  [0.00e+00 1.00e+00 0.00e+00 5.70e+04 2.61e+04 5.00e+00]
#  [0.00e+00 1.00e+00 0.00e+00 2.25e+04 4.00e+04 2.00e+00]
#  [0.00e+00 1.00e+00 0.00e+00 4.60e+04 3.15e+04 4.00e+00]
#  [1.00e+00 0.00e+00 0.00e+00 5.90e+04 2.94e+04 5.00e+00]
#  [1.00e+00 0.00e+00 0.00e+00 5.20e+04 3.20e+04 5.00e+00]
#  [1.00e+00 0.00e+00 0.00e+00 7.20e+04 1.93e+04 6.00e+00]
#  [1.00e+00 0.00e+00 0.00e+00 9.10e+04 1.20e+04 8.00e+00]
#  [0.00e+00 0.00e+00 1.00e+00 6.70e+04 2.20e+04 6.00e+00]
#  [0.00e+00 0.00e+00 1.00e+00 8.30e+04 2.00e+04 7.00e+00]
#  [0.00e+00 0.00e+00 1.00e+00 7.90e+04 2.10e+04 7.00e+00]
#  [0.00e+00 0.00e+00 1.00e+00 5.90e+04 3.30e+04 5.00e+00]]


# How to interpret this 2D Numpy array, which is supposedly representing the dataset?
# Interpreting each value: 
# E.g. 0.00e+00 represents 0.00 x 10^0 = 0, 1.00e+00 represents 1.00 x 10^0 = 1, 6.90e+04 represents 
# 6.90 x 10^4 = 69000, 6.00e^+00 represents 6.00 x 10^0 = 6

# Interpreting the 2D Numpy array's columns:
# These are the corresponding (hidden) column headers of the dataset for each column of this 2D Numpy array:
# Audi | BMW X5 | Mercedez Benz C class class | mileage | sell_price | age
#        [[0.00e+00 1.00e+00 0.00e+00 6.90e+04 1.80e+04 6.00e+00]
#         [0.00e+00 1.00e+00 0.00e+00 3.50e+04 3.40e+04 3.00e+00]
#         [0.00e+00 1.00e+00 0.00e+00 5.70e+04 2.61e+04 5.00e+00]
#         [0.00e+00 1.00e+00 0.00e+00 2.25e+04 4.00e+04 2.00e+00]
#         [0.00e+00 1.00e+00 0.00e+00 4.60e+04 3.15e+04 4.00e+00]
#         [1.00e+00 0.00e+00 0.00e+00 5.90e+04 2.94e+04 5.00e+00]
#         [1.00e+00 0.00e+00 0.00e+00 5.20e+04 3.20e+04 5.00e+00]
#         [1.00e+00 0.00e+00 0.00e+00 7.20e+04 1.93e+04 6.00e+00]
#         [1.00e+00 0.00e+00 0.00e+00 9.10e+04 1.20e+04 8.00e+00]
#         [0.00e+00 0.00e+00 1.00e+00 6.70e+04 2.20e+04 6.00e+00]
#         [0.00e+00 0.00e+00 1.00e+00 8.30e+04 2.00e+04 7.00e+00]
#         [0.00e+00 0.00e+00 1.00e+00 7.90e+04 2.10e+04 7.00e+00]
#         [0.00e+00 0.00e+00 1.00e+00 5.90e+04 3.30e+04 5.00e+00]]



# Now we will need to drop/remove 2 columns the concatenated dummy variable columns, storing binary 
# values, 0 or 1 (aka binary vectors), and the initial dataset, which are (using the '.drop()'
# Pandas library function):
# - The 'car_model' column, which is storing the nominal categorical variables, since we no 
#   longer need it and the nominal categorical variables in the 'car_model' column cannot be 
#   fed into the ML algorithm/model 

# - Any one of the dummy variable columns ('Audi', 'BMW X5' or 'Mercedez Benz C class class' 
#   column) (in this excercise, I will be dropping the 'Mercedez Benz C class class' dummy variable column), storing 
#   binary values, 0 or 1 (aka binary vectors), to prevent a common pitfall when using dummy variables, 
#   known as the Dummy Variable Trap
#   (see the file '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_
#   using_dummy_variables.py' for what a Dummy Variable Trap is)

# Since the 'car_model' column is already automatically dropped by the Scikit-learn's 'OneHotEncoder' class,
# we only need to drop one of the dummy variable columns here.
one_hot_encoded_dataset = np.concatenate([concatenated_dummy_variable_columns_storing_binary_values_with_dataset[:,:2], concatenated_dummy_variable_columns_storing_binary_values_with_dataset[:,3:]], axis=1)
print("Audi | BMW X5 | mileage | sell_price | age")
print(one_hot_encoded_dataset)

# Output:
#    Audi | BMW X5 | mileage | sell_price | age
# [[0.00e+00 1.00e+00 6.90e+04 1.80e+04 6.00e+00]
#  [0.00e+00 1.00e+00 3.50e+04 3.40e+04 3.00e+00]
#  [0.00e+00 1.00e+00 5.70e+04 2.61e+04 5.00e+00]
#  [0.00e+00 1.00e+00 2.25e+04 4.00e+04 2.00e+00]
#  [0.00e+00 1.00e+00 4.60e+04 3.15e+04 4.00e+00]
#  [1.00e+00 0.00e+00 5.90e+04 2.94e+04 5.00e+00]
#  [1.00e+00 0.00e+00 5.20e+04 3.20e+04 5.00e+00]
#  [1.00e+00 0.00e+00 7.20e+04 1.93e+04 6.00e+00]
#  [1.00e+00 0.00e+00 9.10e+04 1.20e+04 8.00e+00]
#  [0.00e+00 0.00e+00 6.70e+04 2.20e+04 6.00e+00]
#  [0.00e+00 0.00e+00 8.30e+04 2.00e+04 7.00e+00]
#  [0.00e+00 0.00e+00 7.90e+04 2.10e+04 7.00e+00]
#  [0.00e+00 0.00e+00 5.90e+04 3.30e+04 5.00e+00]]


# ////////////////////////////////////////////////////////////////////////////////


# Creating the Multiple Variable Linear Regression (MVLR) ML model that we will be testing our 
# One-hot Encoded dataset on (refer to the 'Tutorial 2.1 - Multiple Variable Linear Regression (Supervised 
# Regression Machine Learning Algorithm)' for more about the MVLR ML algorithm)

# Creating a 'MVLR ML model' class object/instance
multiple_variable_linear_regression_ML_model = LinearRegression()



# Training the 'MVLR ML model' class object/instance with the One-hot Encoded dataset

# Getting the independent variables/features, which is all the columns in the 'one_hot_encoded_dataset' except the 
# 'sell_price' column (the dependent variable). Hence, here we dropped/removed the 'sell_price' column, using the 
# code, 'one_hot_encoded_dataset[:,:-1]', which omits the last column of the 'one_hot_encoded_dataset', and taking 
# the  remaining 'one_hot_encoded_dataset' as the independent variables/features

# The dependent variable is the 'sell_price' column in the initial dataset
independent_variables_or_features = np.concatenate([one_hot_encoded_dataset[:,:3], one_hot_encoded_dataset[:,4:]], axis=1)
print(independent_variables_or_features)
multiple_variable_linear_regression_ML_model.fit(independent_variables_or_features, dataset['sell_price'])
# multiple_variable_linear_regression_ML_model.fit(one_hot_encoded_dataset[['Audi', 'BMW X5', 'mileage', 'age']], dataset['sell_price']) works too


# In this tutorial, the trained MVLR ML model's coefficients and intercept/biases might look pretty 
# cryptic due to the OHE ML Encoding Technique, but it works as ML parameters 
# (i.e. coefficients/weights/biases) for the trained MVLR ML model to give accurate predictions

# The '.coef_' attribute of the 'MVLR ML model' class shows the values of the 
# gradient/coefficient of the independent variables/features 'x1', 'x2', 'x3', 'x4' ... 'x?', 'm1', 'm2', 'm3', 'm4'
# ... 'm?' 
# respectively, in the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + m4*x4 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [-2.45354074e+03 -6.73820733e+03 -3.70122094e-01 -1.33245363e+03]

# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file in the 'Tutorial 2.1 - Multiple Variable Linear Regression 
# (Supervised Regression Machine Learning Algorithm)' folder), 'b', in the mathematical equation representing the MVLR ML 
# algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + m4*x4 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.intercept_)                     # output: 58976.625968552275




# Making predictions with the 'MVLR ML model' class object/instance, which is trained with the One-hot Encoded dataset

# In this context, 'x1' represents the 'Audi' independent variable/feature, 'x2' represents the 'BMW X5' 
# independent variable/feature, 'x3' represents the 'mileage' independent variable/feature, 'x4' represents the
# 'age' independent variable/feature


# 1) Predict sell price of a Mercedez Benz C class that is, 4 years old with mileage 45000
# To get a prediction of a 'sell_price' (dependent variable) for a car of the brand 'Mercedez Benz C class', the 
# value of 'x1' (representing 'Audi' independent variable/feature) must be '0', while the value of and 'x2' 
# (representing 'BMW X5' independent variable/feature) must also be '0'
print(multiple_variable_linear_regression_ML_model.predict([[0,0,45000,4]]))          # Output: 36991.31721062

# 2) Predict sell price of a BMW X5 that is, 7 years old with mileage 86000
# To get a prediction of a 'sell_price' (dependent variable) for a car of the brand 'BMW X5', the value of 'x1' 
# (representing 'Audi' independent variable/feature) must be '0', while the value of and 'x2' (representing 'BMW X5' 
# independent variable/feature) must be '1'
print(multiple_variable_linear_regression_ML_model.predict([[0,1,86000,7]]))          # Output: 11080.74313218


# ///////////////////////////////////////////////////////////////////////////////////


# 3) Tell me the score (accuracy) of your Linear Regression ML model. (Hint: use the '.score()' Instance Method
#    of the 'LinearRegression' class)

# Apart from the the MVLR ML algorithm class giving us access to the '.fit()' and '.predict()' Instance Methods and the
# '.coef_' and '.intercept_' attributes, the MVLR ML algorithm class also gives us access to the '.score()' Instance
# Method, which allows us to calculate how accurate the MVLR ML model is:
print(multiple_variable_linear_regression_ML_model.score(independent_variables_or_features, dataset['sell_price']))

# Output: 0.9417050937281082
# This means that the MVLR ML model is 94% accurate 




# My answers are all correct.