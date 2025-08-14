# Question 1:
# Given the hiring salary statistics for a firm, in the dataset 'hiring_salary', containing the 3 (multiple) 
# variables, 
#   -> experience of candidate 
#   -> written test score
#   -> personal interview score. 

# Based on these 3 (multiple) variables, the Human Resource (HR) department of the firm will decide the salary. 
# Given this dataset, you need to use the Multiple Variable Linear Regression Supervised learning Regression 
# Machine Learning (ML) algorithm to help the HR department decide the hiring salaries for future candidates. 

# Using the Multiple Variable Linear Regression Supervised learning Regression Machine Learning (ML) algorithm,
# predict salaries for following candidates:
#   1. A candidate with 2 years experience, 9 for the written test score, 6 for the interview score
#   2. A candidate with 12 years experience, 10 for the written test score, 10 for the interview score

# (Hint: you may need to use the external Python module 'word2number' (documentation link: 
# https://pypi.org/project/word2number/) to convert strings (of English writing of the numbers) to the corresponding 
# integer numbers during the cleaning and preparing of the dataset prior to training the Multiple Variable Linear 
# Regression (MVLR) Supervised learning Regression Machine Learning (ML) algorithm)


import pandas as pd
import math
from word2number import w2n
from sklearn.linear_model import LinearRegression

# Cleaning and preparing the 'hiring_salary.csv' dataset:

# Reading the 'hiring_salary.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv('hiring_salary.csv')


# As we can see, the 'hiring_salary.csv' dataset is messy with the following errors:
# - inappropriate datatype input in the 'experience' column that is strings (of English writing of the numbers), rather 
#   than the expected datatype input of integers
# - missing NaN/NA/Null values/cells in the 'experience' and 'test_score(out of 10)' columns 

# Hence, we will need to clean and prepare this dataset, by handling the inappropriate datatype input in the 'experience' 
# column in the dataset that is strings (of English writing of the numbers), rather than the expected datatype input of 
# integers and the missing NaN/NA/Null values/cells in the 'experience' and 'test_score(out of 10)' columns in the dataset
                #   experience  test_score(out of 10)  interview_score(out of 10)  salary($)
                # 0        NaN                    8.0                           9      50000
                # 1        NaN                    8.0                           6      45000
                # 2       five                    6.0                           7      60000
                # 3        two                   10.0                          10      65000
                # 4      seven                    9.0                           6      70000
                # 5      three                    7.0                          10      62000
                # 6        ten                    NaN                           7      72000
                # 7     eleven                    7.0                           8      80000
print(dataset)


# To handle the inappropriate datatype input in the 'experience' column in the dataset that is strings
# (of English writing of the numbers), rather than the expected datatype input of integers, we will convert 
# the strings (of English writing of the numbers) into the corresponding integer numbers using the 
# 'word_to_num()' function in the 'w2n' sub-module of the 'word2number' external Python module
temp_list_to_hold_the_converted_strings = []

for i in dataset['experience']:
    print(type(i))      # Using this, I figured out that the NaN/NA/Null value/cell are of the float datatype
                        # apparently                         

    if isinstance(i, float) is False:
        print(w2n.word_to_num(i))
        temp_list_to_hold_the_converted_strings.append(w2n.word_to_num(i))
    else:
        temp_list_to_hold_the_converted_strings.append(None)

print(temp_list_to_hold_the_converted_strings)

dataset.loc[:, 'experience'] = temp_list_to_hold_the_converted_strings

print(dataset)


# After the error of the inappropriate datatype input in the 'experience' column in the dataset that is strings
# (of English writing of the numbers), rather than the expected datatype input of integers, is solved, now we 
# will use the median of all the values in the 'experience' and 'test_score(out of 10)' columns respectively 
# in the dataset to 'fill in' the missing NaN/NA/Null value/cell in the 'experience' and 'test_score(out of 10)' 
# columns respectively in the dataset, to handle the missing NaN/NA/Null value/cell in the 'experience' and 
# 'test_score(out of 10)' columns respectively in the dataset.

# Finding the median of all the values in the 'experience' column in the dataset
median_number_of_years_of_experience = dataset['experience'].median()
print(median_number_of_years_of_experience)                         # output: 6.0

# Finding the median of all the values in the 'test_score(out of 10)' column in the dataset
median_test_score = dataset['test_score(out of 10)'].median()
print(median_test_score)                                            # output: 8.0


# Rounding down the value of the median (using the '.median()' Pandas function) of all the values in the 
# 'experience' and 'test_score(out of 10)' columns respectively in the dataset that we will be using as a 
# 'fill in' of this missing NaN/NA/Null value/cell in the 'experience' column in the dataset.

# Rounding down the median of all the values in the 'experience' column in the dataset
always_rounded_down_median_number_of_years_of_experience = math.floor(median_number_of_years_of_experience)
print(always_rounded_down_median_number_of_years_of_experience)     # output: 6

# Rounding down the median of all the values in the 'experience' column in the dataset
always_rounded_down_median_test_score = math.floor(median_test_score)
print(always_rounded_down_median_test_score)                        # output: 8


# Using the '.fillna()' Pandas function to 'fill in' any missing NaN/NA/Null value/cell in a column, and in this
# case, the missing NaN/NA/Null value/cell in the 'experience' and 'test_score(out of 10)' columns respectively 
# in the dataset with the always rounded down median value stored in the 
# 'always_rounded_down_median_number_of_years_of_experience' and 'always_rounded_down_median_test_score' 
# variables respectively

# 'Filling in' the missing NaN/NA/Null value/cell in the 'experience' column in the dataset with the always
# rounded down median value stored in the 'always_rounded_down_median_number_of_years_of_experience' variable
dataset['experience'] = dataset['experience'].fillna(always_rounded_down_median_number_of_years_of_experience)

# 'Filling in' the missing NaN/NA/Null value/cell in the 'test_score(out of 10)' column in the dataset with the 
# always rounded down median value stored in the 'always_rounded_down_median_test_score' variable
dataset['test_score(out of 10)'] = dataset['test_score(out of 10)'].fillna(always_rounded_down_median_test_score)


# Here is the cleaned and prepared dataset, with the missing NaN/NA/Null value/cell in the 'experience' and 
# 'test_score(out of 10)' columns respectively in the dataset now handled, and is ready to be used to train 
# the Multiple Variable Linear Regression (MVLR) ML algorithm to become a MVLR ML model.
print(dataset)


# ////////////////////////////////////////////////////////////////////////////////////////////////


# Creating a 'Multiple Variable Linear Regression (MVLR) ML model' class object/instance
multiple_variable_linear_regression_ML_model = LinearRegression()


# Training the 'MVLR ML model' class object/instance
multiple_variable_linear_regression_ML_model.fit(dataset[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], dataset['salary($)'])
print(multiple_variable_linear_regression_ML_model)


# The '.coef_' attribute of the 'MVLR ML model' class shows the values of the 
# gradient/coefficient of the independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1', 'm2', 'm3' ... 'm?' respectively, 
# in the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [2813.00813008 1333.33333333 2926.82926829]


# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file), 'b', in the mathematical equation representing the MVLR ML 
# algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.intercept_)                     # output: 11869.918699186957


# Making predictions with the 'MVLR ML model' class object/instance, predicting the hiring salary, using the MVLR Supervised 
# learning Regression Machine Learning (ML) algorithm, of:
#   1. A candidate with 2 years experience, 9 for the written test score, 6 for the interview score
print(multiple_variable_linear_regression_ML_model.predict([[2, 9, 6]]))           # output: [47056.91056911]

# Making predictions with the 'MVLR ML model' class object/instance, predicting the hiring salary, using the MVLR Supervised 
# learning Regression Machine Learning (ML) algorithm, of:
#   2. A candidate with 12 years experience, 10 for the written test score, 10 for the interview score
print(multiple_variable_linear_regression_ML_model.predict([[12, 10, 10]]))        # output: [88227.64227642]


# /////////////////////////////////////////////////////////////////////////////////////////////////


# Some key observations from all this:

# So how is this predicted value/output made by the 'Multiple Variable Linear Regression (MVLR) ML model' class 
# object/instance?
# Essentially, looking at the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  , since:
# - the '.coef_' attribute of the 'MVLR ML model' class shows the value of the gradient/coefficient of the independent 
#   variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1' ('2813.00813008'), 'm2' ('1333.33333333'), 'm3' ('2926.82926829')
#   respectively 
# - the input 'X' parameter of this '.predict()' Instance Method of the 'MVLR ML model' class is the independent 
#   variables/features, 'x1' ('2'/'12'), 'x2' ('9'/'10'), 'x3' ('6'/'10')
# - the '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
#   '1. What_is_Multiple_Variable_Linear_Regression.txt' file), 'b' ('11869.918699186957')

# And since the output of this '.predict()' Instance Method of the 'MVLR ML model' class is the dependent variable, 'y' 
# ('47056.91056911'/'88227.64227642'), to prove that 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b' is the mathematical 
# equation representing the MVLR ML algorithm, we can substitude the values of of 'm1', 'm2', 'm3', 'x1', 'x2', 'x3' and 
# 'b' into the 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b' mathematical equation to find the same value of 'y', which is 
# also the predicted value/output made by the 'MVLR ML model' class object/instance shown via the '.predict()' Instance Method 
# of the 'MVLR ML model' class

# For '1. A candidate with 2 years experience, 9 for the written test score, 6 for the interview score''s MVLR ML model's
# prediction
#          'm1'     * 'x1' +     'm2'      *  'x2' +      'm3'     * 'x3' +       'b'                  =   y
print(2813.00813008 *  2   + 1333.33333333 *   9   + 2926.82926829 *  6   + 11869.918699186957)      # = 47056.91056911   

# For '2. A candidate with 12 years experience, 10 for the written test score, 10 for the interview score''s MVLR ML model's
# prediction
#          'm1'     * 'x1' +     'm2'      *  'x2' +      'm3'     * 'x3' +       'b'                  =   y
print(2813.00813008 *  12  + 1333.33333333 *   10  + 2926.82926829 *  10  + 11869.918699186957)      # = 88227.64227642  

