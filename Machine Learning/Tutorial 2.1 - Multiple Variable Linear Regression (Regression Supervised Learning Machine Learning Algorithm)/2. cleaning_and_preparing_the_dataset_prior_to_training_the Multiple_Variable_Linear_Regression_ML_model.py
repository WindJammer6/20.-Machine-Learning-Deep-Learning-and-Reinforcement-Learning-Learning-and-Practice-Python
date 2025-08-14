# Cleaning and preparing the dataset in Computer Science is vital in not only in Data Analysis, but also in Machine 
# Learning (ML), such as when we train Machine Learning (ML) algorithms to become ML models, to ensure that we obtain 
# accurate and reliable results from the programs.

# In this file, we will be cleaning and preparing the dataset that we will be using to train the Multiple Variable 
# Linear Regression (MVLR) ML algorithm to become a Multiple Variable Linear Regression model.

# There are many ways to clean and prepare a dataset for data analysis/modelling of a ML algorithm. See my other 
# Data Analysis (Personal Projects) Github repositories on my Github profile (https://github.com/WindJammer6), 
# '8.-Star-Wars-Data-Analysis-Python' and '9.-Employee-Exit-Data-Analysis-Python' for some other ways to clean and 
# prepare a dataset that I have done in those Github repositories. 

# However, in this file, we will only specifically focus on the cleaning and preperation of the dataset's task of 
# handling missing NaN/NA/Null values/cells in the dataset.


import pandas as pd
import math

# Reading the 'house_prices_with_multiple_independent_variables.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv('house_prices_with_multiple_independent_variables.csv')


# As we can see, the 'house_prices_with_multiple_independent_variables.csv' dataset is messy and has a missing NaN/NA/Null 
# value/cell] in the 'bedrooms' column in the dataset. Hence, we will need to clean and prepare this dataset, by 
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

# The '.round()' built-in Python function, will not work in this case if we want to always round down the median 
# value, regardless of the float value. This is because the '.round()' built-in Python function will round the
# float value up if its decimal values are larger than '.5', and if the float value's decimal values are exactly 
# '.5', then the '.round()' built-in Python function will round the float value to the nearest even integer 
# number (which can be round up or down, depending on the float value).
always_rounded_down_median_bedrooms = math.floor(median_bedrooms)
print(always_rounded_down_median_bedrooms)                      # output: 3        


# Using the '.fillna()' Pandas function to 'fill in' any missing NaN/NA/Null value/cell in a column, and in this
# case, the missing NaN/NA/Null value/cell in the 'bedrooms' column in the dataset with the always rounded down
# median value stored in the 'always_rounded_down_median_bedrooms' variable

# (Note: You cannot just do 'dataset['bedrooms'].fillna(always_rounded_down_median_bedrooms)'. You need to 
# rememeber to 'put back' the new, modified/cleaned and prepared 'bedrooms' column, with the 'filled' missing 
# NaN/NA/Null value/cell with the always rounded down median value, back into the dataset, like so)
dataset['bedrooms'] = dataset['bedrooms'].fillna(always_rounded_down_median_bedrooms)

# Here is the cleaned and prepared dataset, with the missing NaN/NA/Null value/cell in the 'bedrooms' column in 
# the dataset now handled, and is ready to be used to train the MVLR ML algorithm to become a MVLR model.
print(dataset)



