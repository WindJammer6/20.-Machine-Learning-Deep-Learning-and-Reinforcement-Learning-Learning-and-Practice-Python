import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

# Creating an One-hot Encoded dataset of nominal categorical variables using Scikit-learn's 'OneHotEncoder' class
# (Note: The Scikit-learn's 'OneHotEncoder' class does the exact same functionality as the code in the file 
# '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_using_dummy_variables.py')

# Reading the 'house_prices_with_nominal_categorial_variables.csv' dataset from the CSV file using the Pandas 
# library
dataset = pd.read_csv("house_prices_with_nominal_categorial_variables.csv")
print(dataset)



# Not very important, just due to Scikit-learn documentation, in order to apply the Scikit-learn's 'OneHotEncoder' 
# class, there is a need to apply the Scikit-learn 'ColumnTransformer' class beforehand whcih helps to specify 
# which column in a dataset to apply the Scikit-learn's 'OneHotEncoder' class to 
columntransformer_with_onehotencoder = ColumnTransformer(transformers=[("house_location", OneHotEncoder(), [0])], remainder='passthrough')



# Using the '.fit_transform()' Instance method of the Scikit-learn's 'OneHotEncoder' class to 'fit' the nominal
# categorical variables, in the 'house_location' column, into the 'OneHotEncoder' to do One-hot Encoding on the 
# nominal categorical variables. The Scikit-learn's 'OneHotEncoder' class 'fits' the nominal categorical 
# variables in ASCII order.

# The 'fitted' nominal categorical variables, in the 'house_location' column, into the 'OneHotEncoder' is then
# assigned back into the initial dataset
concatenated_dummy_variable_columns_storing_binary_values_with_dataset = columntransformer_with_onehotencoder.fit_transform(dataset)
print(concatenated_dummy_variable_columns_storing_binary_values_with_dataset)

# Output: 
# [[1.00e+00 0.00e+00 0.00e+00 2.60e+03 5.50e+05]
#  [1.00e+00 0.00e+00 0.00e+00 3.00e+03 5.65e+05]
#  [1.00e+00 0.00e+00 0.00e+00 3.20e+03 6.10e+05]
#  [1.00e+00 0.00e+00 0.00e+00 3.60e+03 6.80e+05]
#  [1.00e+00 0.00e+00 0.00e+00 4.00e+03 7.25e+05]
#  [0.00e+00 1.00e+00 0.00e+00 2.60e+03 5.75e+05]
#  [0.00e+00 1.00e+00 0.00e+00 2.90e+03 6.00e+05]
#  [0.00e+00 1.00e+00 0.00e+00 3.10e+03 6.20e+05]
#  [0.00e+00 1.00e+00 0.00e+00 3.60e+03 6.95e+05]
#  [0.00e+00 0.00e+00 1.00e+00 2.60e+03 5.85e+05]
#  [0.00e+00 0.00e+00 1.00e+00 2.80e+03 6.15e+05]
#  [0.00e+00 0.00e+00 1.00e+00 3.30e+03 6.50e+05]
#  [0.00e+00 0.00e+00 1.00e+00 3.60e+03 7.10e+05]]


# How to interpret this 2D Numpy array, which is supposedly representing the dataset?
# Interpreting each value: 
# E.g. 0.00e+00 represents 0.00 x 10^0 = 0, 1.00e+00 represents 1.00 x 10^0 = 1, 2.60e+03 represents 
# 2.60 x 10^3 = 2600, 7.10e^+05 represents 7.10 x 10^5 = 71000

# Interpreting the 2D Numpy array's columns:
# These are the corresponding (hidden) column headers of the dataset for each column of this 2D Numpy array:
# monroe_township | robinsville | west windsor | area | price
#   [[1.00e+00 0.00e+00 0.00e+00 2.60e+03 5.50e+05]
#    [1.00e+00 0.00e+00 0.00e+00 3.00e+03 5.65e+05]
#    [1.00e+00 0.00e+00 0.00e+00 3.20e+03 6.10e+05]
#    [1.00e+00 0.00e+00 0.00e+00 3.60e+03 6.80e+05]
#    [1.00e+00 0.00e+00 0.00e+00 4.00e+03 7.25e+05]
#    [0.00e+00 1.00e+00 0.00e+00 2.60e+03 5.75e+05]
#    [0.00e+00 1.00e+00 0.00e+00 2.90e+03 6.00e+05]
#    [0.00e+00 1.00e+00 0.00e+00 3.10e+03 6.20e+05]
#    [0.00e+00 1.00e+00 0.00e+00 3.60e+03 6.95e+05]
#    [0.00e+00 0.00e+00 1.00e+00 2.60e+03 5.85e+05]
#    [0.00e+00 0.00e+00 1.00e+00 2.80e+03 6.15e+05]
#    [0.00e+00 0.00e+00 1.00e+00 3.30e+03 6.50e+05]
#    [0.00e+00 0.00e+00 1.00e+00 3.60e+03 7.10e+05]]


# How does the 'Scikit-learn's 'OneHotEncoder' class create the dummy variable columns?
# 1. The Scikit-learn's 'OneHotEncoder' class (with the help of the Scikit-learn's 'ColumnTransformer' class), first 
#    scans through the 'house_location' column, which is storing the nominal categorical variables, and identifies 
#    all the distinct nominal categorical variables ('monroe township', 'robinsville' and 'west windsor')
# 2. The Scikit-learn's 'OneHotEncoder' class then sorts these distinct nominal categorical variables in
#    ASCII order (i.e. If you have the nominal categorical variables: 'monroe township', 'west windsor' and
#    'robinsville', the Scikit-learn's 'OneHotEncoder' class will sort them in the order of: 'monroe 
#    township', 'robinsville' and 'west windsor', because the ASCII order of the first letter of each nominal 
#    categorical variables is as follows: 'm' < 'r' < 'w'). Then, it will create the dummy variable columns with 
#    the nominal categorical variables as headers in the ASCII order, which will be 'monroe township', 'robinsville'
#    then 'west windsor'.
# 3. Then based on the order of the each item in the dataset, it will assign a value of either 0 or 1, 
#    depending if the item does/does not belongs to that category variable.
# 4. IMPORTANT! It then concatenates the dummy variable columns TO THE FRONT of the initial dataset, and deletes
#    the 'house_location' column, which is storing the nominal categorical variables.




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
#   (see the file '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_
#   using_dummy_variables.py' for what a Dummy Variable Trap is)

# Since the 'house_location' column is already automatically dropped by the Scikit-learn's 'OneHotEncoder' class,
# we only need to drop one of the dummy variable columns here.

# After dropping these 2 columns, this concatenated dummy variable columns, storing binary values, 0 or 1 
# (aka binary vectors), and the initial dataset, is now a One-hot Encoded dataset

# (Note: Using the Numpy's '.concatenate()' function instead of Pandas' '.concat()' function since
#  unlike the '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_using_dummy_
#  variable.py' file, where the 'concatenated_dummy_variable_columns_storing_binary_values_with_dataset' variable 
#  is storing a Pandas dataframe, in this file, the 'concatenated_dummy_variable_columns_storing_binary_values_
#  with_dataset' variable is storing a 2D Numpy array instead (Panda's '.concat()' function won't work on 2D numpy 
#  arrays)
one_hot_encoded_dataset = np.concatenate([concatenated_dummy_variable_columns_storing_binary_values_with_dataset[:,:2], concatenated_dummy_variable_columns_storing_binary_values_with_dataset[:,3:]], axis=1)
print("monroe township | robinsville | area | price")
print(one_hot_encoded_dataset)

# Output:
# monroe township | robinsville | area | price
# [[1.00e+00 0.00e+00 2.60e+03 5.50e+05]
#  [1.00e+00 0.00e+00 3.00e+03 5.65e+05]
#  [1.00e+00 0.00e+00 3.20e+03 6.10e+05]
#  [1.00e+00 0.00e+00 3.60e+03 6.80e+05]
#  [1.00e+00 0.00e+00 4.00e+03 7.25e+05]
#  [0.00e+00 1.00e+00 2.60e+03 5.75e+05]
#  [0.00e+00 1.00e+00 2.90e+03 6.00e+05]
#  [0.00e+00 1.00e+00 3.10e+03 6.20e+05]
#  [0.00e+00 1.00e+00 3.60e+03 6.95e+05]
#  [0.00e+00 0.00e+00 2.60e+03 5.85e+05]
#  [0.00e+00 0.00e+00 2.80e+03 6.15e+05]
#  [0.00e+00 0.00e+00 3.30e+03 6.50e+05]
#  [0.00e+00 0.00e+00 3.60e+03 7.10e+05]]


# Differences between the output of the 'one_hot_encoded_dataset' variable in this file and the 
# '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_using_dummy_variable.py' file:
# 1. In this file, the 'one_hot_encoded_dataset' variable is storing the initial dataset as a 2D Numpy array 
#    (without the initial dataset's column headers), while in the '2. manually_creating_a_One-hot_Encoded_dataset_
#    of_nominal_categorical_variables_using_dummy_variable.py' file the 'one_hot_encoded_dataset' variable is 
#    storing the initial dataset as a Pandas dataframe (with the initial dataset's column headers)

# 2. In this file, the positioning of the One-hot Encoded dataset columns of the remaining nominal categorical 
#    variables of the 'house_location' column, 'monroe township' and 'robinsville' dummy columns are now located 
#    at the front columns of the dataset, because the 'Scikit-learn's 'OneHotEncoder' class concatenates the 
#    created dummy variable columns to the front of the initial dataset,
#                    monroe_township | robinsville | area | price
#                       [[1.00e+00 0.00e+00 2.60e+03 5.50e+05]
#                        [1.00e+00 0.00e+00 3.00e+03 5.65e+05]
#                        [1.00e+00 0.00e+00 3.20e+03 6.10e+05]
#                        [1.00e+00 0.00e+00 3.60e+03 6.80e+05]
#                        [1.00e+00 0.00e+00 4.00e+03 7.25e+05]
#                        [0.00e+00 1.00e+00 2.60e+03 5.75e+05]
#                        [0.00e+00 1.00e+00 2.90e+03 6.00e+05]
#                        [0.00e+00 1.00e+00 3.10e+03 6.20e+05]
#                        [0.00e+00 1.00e+00 3.60e+03 6.95e+05]
#                        [0.00e+00 0.00e+00 2.60e+03 5.85e+05]
#                        [0.00e+00 0.00e+00 2.80e+03 6.15e+05]
#                        [0.00e+00 0.00e+00 3.30e+03 6.50e+05]
#                        [0.00e+00 0.00e+00 3.60e+03 7.10e+05]]

#    but in the '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_using_dummy_
#    variable.py' file, the positioning of the One-hot Encoded dataset columns of the 'house_location' column, 
#    'monroe township' and 'robinsville' dummy columns are located at the back columns of the dataset, because
#    the Pandas' '.concat()' function concatenates the created dummy variable columns to the back of the initial 
#    dataset,
#                       area   price  monroe township  robinsville
#                   0   2600  550000                1            0
#                   1   3000  565000                1            0
#                   2   3200  610000                1            0
#                   3   3600  680000                1            0
#                   4   4000  725000                1            0
#                   5   2600  575000                0            1
#                   6   2900  600000                0            1
#                   7   3100  620000                0            1
#                   8   3600  695000                0            1
#                   9   2600  585000                0            0
#                   10  2800  615000                0            0
#                   11  3300  650000                0            0
#                   12  3600  710000                0            0

#    THE POSITIONING OF THE COLUMNS OF THE ONE-HOT ENCODED DATASET IS CRUCIAL, as it affects the value of the 
#    coefficients of the Machine Learning (ML) algorithm which the One-hot Encoded dataset is being used to
#    be trained with! (refer to the 'Training the 'MVLR ML model' class object/instance with the One-hot 
#    Encoded dataset' section below). You may notice that the section below 'Creating the Multiple Variable 
#    Linear Regression (MVLR) ML model' differs slightly in this file with the same section in the 
#    '2. manually_creating_a_One-hot_Encoded_dataset_of_nominal_categorical_variables_using_dummy_variables.py'
#    file.


# ////////////////////////////////////////////////////////////////////////////////


# Creating the Multiple Variable Linear Regression (MVLR) ML model that we will be testing our 
# One-hot Encoded dataset on (refer to the 'Tutorial 2.1 - Multiple Variable Linear Regression (Supervised 
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
# 'price' column (the dependent variable). Hence, here we dropped/removed the 'price' column, using the code, 
# 'one_hot_encoded_dataset[:,:-1]', which omits the last column of the 'one_hot_encoded_dataset', and taking the  
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
#        the code 'one_hot_encoded_dataset[['monroe township', 'robinsville', 'area']]' matters, as in the mathematical 
#        equation representing the Multiple Variable Linear Regression (MVLR) ML algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... 
#        + m?*x? + b', the first column name in the code 'one_hot_encoded_dataset[['monroe township', 'robinsville', 'area']]' 
#        will be 'x1', the second column name will be 'x2', and so on... In the mathematical equation for the MVLR ML 
#        algorithm in this context, aka the Machine Learning (ML) ML model for this context, it is:
#                    'y'            'x1'              'x2'          'x3' 
#                   price = m1*monroe township + m2*robinsville + m3*area + b

#        (This only applies for the MVLR ML algorithm, and not for the Single Variable Linear Regression (SVLR) ML algorithm,
#        since the SVLR ML algorithm only has 1 column/independent variable/feature):
independent_variables_or_features = one_hot_encoded_dataset[:,:-1]
print(independent_variables_or_features)
multiple_variable_linear_regression_ML_model.fit(independent_variables_or_features, dataset['price'])
# multiple_variable_linear_regression_ML_model.fit(one_hot_encoded_dataset[['monroe township', 'robinsville', 'area']], dataset['price']) works too



# In this tutorial, the trained MVLR ML model's coefficients and intercept/biases might look pretty 
# cryptic due to the OHE ML Encoding Technique, but it works as ML parameters 
# (i.e. coefficients/weights/biases) for the trained MVLR ML model to give accurate predictions

# The '.coef_' attribute of the 'MVLR ML model' class shows the values of the 
# gradient/coefficient of the independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1', 'm2', 'm3' ... 'm?' 
# respectively, in the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  

# (Note: The multiple coefficient values of the independent variables/features, 'm1', 'm2', 'm3' ... 'm?', will be displayed
#        in the output in order, in a Python list, for example, '[-40013.97548914 -14327.56396474    126.89744141]', where 
#        '-40013.97548914' will be 'm1', '-14327.56396474' will be 'm2', and '126.89744141' will be 'm3')
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [-40013.97548914 -14327.56396474    126.89744141]

# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file in the 'Tutorial 2.1 - Multiple Variable Linear Regression 
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
# Instance Method of the 'MVLR ML model' class will need to be in order, in a 2D array, where '1' will be 'x1', '0' will
# be 'x2', and '2800' will be 'x3', and that they cannot be in a list/1D array like, '[2800,0,1]', but must be in a 2D 
# array like, '[[1,0,2800]]'


# In this context, 'x1' represents the 'monroe township' independent variable/feature, 'x2' represents the 'robinsville' 
# independent variable/feature, 'x3' represents the 'area' independent variable/feature

# To get a prediction of a 'price' (dependent variable) for a house in 'monroe township', the value of 'x1' (representing
# 'monroe township' independent variable/feature) must be '1', while the value of and 'x2' (representing 'robinsville' 
# independent variable/feature) must be '0'
print(multiple_variable_linear_regression_ML_model.predict([[1,0,3000]]))          # Output: 590468.71640508
# To get a prediction of a 'price' (dependent variable) for a house in 'robinsville', the value of 'x2' (representing
# 'robinsville' independent variable/feature) must be '1', while the value of and 'x1' (representing 'monroe township' 
# independent variable/feature) must be '0'
print(multiple_variable_linear_regression_ML_model.predict([[0,1,2800]]))          # Output: 590775.63964739
# To get a prediction of a 'price' (dependent variable) for a house in 'west windsor', the value of 'x1' (representing
# 'monroe township' independent variable/feature) must be '0', and the value of and 'x2' (representing 'robinsville' 
# independent variable/feature) must also be '0'
print(multiple_variable_linear_regression_ML_model.predict([[0,0,3400]]))          # Output: 681241.66845839



# Apart from the the MVLR ML algorithm class giving us access to the '.fit()' and '.predict()' Instance Methods and the
# '.coef_' and '.intercept_' attributes, the MVLR ML algorithm class also gives us access to the '.score()' Instance
# Method, which allows us to calculate how accurate the MVLR ML model is:
print(multiple_variable_linear_regression_ML_model.score(independent_variables_or_features, dataset['price']))

# Output: 0.9573929037221872
# This means that the MVLR ML model is 95% accurate 



