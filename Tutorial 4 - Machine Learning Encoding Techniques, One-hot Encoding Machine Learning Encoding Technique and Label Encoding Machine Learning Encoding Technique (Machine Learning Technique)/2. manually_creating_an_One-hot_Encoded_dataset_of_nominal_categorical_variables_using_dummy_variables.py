import pandas as pd

# Manually creating an One-hot Encoded dataset of nominal categorical variables using dummy variables 

# What are dummy variables?
# Dummy variables are extra variables used to represent categorical variables, particularly nominal 
# categorical variables, in a numerical format that can be used by ML algorithms/models. 

# In the context of the OHE ML Encoding Technique, the OHE ML Encoding Technique creates dummy variable
# columns, storing binary values, 0 or 1 (aka binary vectors), with 1 representing that the item belongs 
# to that category variable, and 0 representing that the item does not belongs to that category variable
# E.g.
#    house_location     area   price     monroe township   robinsville   west windsor
#    monroe township  | 2600 | 550000 |               1 |            0 |            0
#    monroe township  | 3000 | 565000 |               1 |            0 |            0
#    monroe township  | 3200 | 610000 |               1 |            0 |            0
#    monroe township  | 3600 | 680000 |               1 |            0 |            0
#    monroe township  | 4000 | 725000 |               1 |            0 |            0
#     robinsville     | 2600 | 575000 |               0 |            1 |            0
#     robinsville     | 2900 | 600000 |               0 |            1 |            0
#     robinsville     | 3100 | 620000 |               0 |            1 |            0
#     robinsville     | 3600 | 695000 |               0 |            1 |            0
#     west windsor    | 2600 | 585000 |               0 |            0 |            1
#     west windsor    | 2800 | 615000 |               0 |            0 |            1
#     west windsor    | 3300 | 650000 |               0 |            0 |            1
#     west windsor    | 3600 | 710000 |               0 |            0 |            1
#                           ('house_location' column is storing 
#                            nominal categorical variables)

# In this example, the 'monroe township', 'robinsville', and 'west winsor' columns are the dummy
# variable columns, storing binary values, 0 or 1 (aka binary vectors).



# Reading the 'house_prices_with_nominal_categorial_variables.csv' dataset from the CSV file using the 
# Pandas library
dataset = pd.read_csv("house_prices_with_nominal_categorial_variables.csv")
print(dataset)



# Creating the dummy variable columns, storing binary values, 0 or 1 (aka binary vectors) using the 
# '.get_dummies()' Pandas Python library function

# How does the '.get_dummies()' Pandas Python library function create the dummy variable columns?
# 1. The '.get_dummies()' Pandas Python library function first scans through the 'house_location' column, which is 
#    storing the nominal categorical variables, and identifies all the distinct nominal categorical variables
#    ('monroe township', 'robinsville' and 'west windsor')
# 2. The '.get_dummies()' Pandas Python library function then sorts these distinct nominal categorical variables in
#    ASCII order (i.e. If you have the nominal categorical variables: 'monroe township', 'west windsor' and
#    'robinsville', the '.get_dummies()' Pandas Python library function will sort them in the order of: 'monroe 
#    township', 'robinsville' and 'west windsor', because the ASCII order of the first letter of each nominal 
#    categorical variables is as follows: 'm' < 'r' < 'w'). Then, it will create the dummy columns with the nominal
#    categorical variables as headers in the ASCII order, which will be 'monroe township', 'robinsville' then 
#    'west windsor'.
# 3. Then based on the order of the each item in the dataset, it will assign a value of either 0 or 1, 
#    depending if the item does/does not belongs to that category variable
dummy_variable_columns_storing_binary_values = pd.get_dummies(dataset['house_location'])
# dummy_variable_columns_storing_binary_values = pd.get_dummies(dataset.house_location) works too
print(dummy_variable_columns_storing_binary_values)

# Output:
#     monroe township  robinsville  west windsor
# 0                 1            0             0
# 1                 1            0             0
# 2                 1            0             0
# 3                 1            0             0
# 4                 1            0             0
# 5                 0            1             0
# 6                 0            1             0
# 7                 0            1             0
# 8                 0            1             0
# 9                 0            0             1
# 10                0            0             1
# 11                0            0             1
# 12                0            0             1



# Concatenating the dummy variable columns, storing binary values, 0 or 1 (aka binary vectors) with 
# the initial dataset using the '.concat()' Pandas library function
concatenated_dummy_variable_columns_storing_binary_values_with_dataset = pd.concat([dataset, dummy_variable_columns_storing_binary_values], axis='columns')
print(concatenated_dummy_variable_columns_storing_binary_values_with_dataset)

# Output:
#      house_location  area   price  monroe township  robinsville  west windsor
# 0   monroe township  2600  550000                1            0             0
# 1   monroe township  3000  565000                1            0             0
# 2   monroe township  3200  610000                1            0             0
# 3   monroe township  3600  680000                1            0             0
# 4   monroe township  4000  725000                1            0             0
# 5       robinsville  2600  575000                0            1             0
# 6       robinsville  2900  600000                0            1             0
# 7       robinsville  3100  620000                0            1             0
# 8       robinsville  3600  695000                0            1             0
# 9      west windsor  2600  585000                0            0             1
# 10     west windsor  2800  615000                0            0             1
# 11     west windsor  3300  650000                0            0             1
# 12     west windsor  3600  710000                0            0             1




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

#   What is the Dummy Variable Trap?
#   The Dummy Variable Trap refers to a scenario in which dummy variables created from a categorical 
#   variable are highly correlated, leading to multicollinearity in regression models. This issue arises 
#   when one of the dummy variables can be perfectly predicted from the others, causing problems in 
#   parameter estimation.
#   Illustrating the problem of the Dummy Variable Trap:
#        marital_status | age | income
#            single     |  25 |  45000
#            married    |  28 |  52000
#           divorced    |  32 |  58000
#            single     |  35 |  62000
#            married    |  40 |  68000
#            single     |  27 |  75000
#            married    |  30 |  81000
#           divorced    |  34 |  89000
#            single     |  37 |  92000
#            married    |  41 |  97000
#           divorced    |  29 | 102000
#            single     |  33 | 108000
#            married    |  38 | 115000

#   Let's say we have this dataset, where the 'martial_status' column is storing the nominal categorical variables,
#   where the 'single' and 'divorced' categorical variables are highly correlated, in the sense that a person that 
#   is 'divorced' will also be 'single'. When we create the dummy variables, storing binary values, 0 or 1 (aka 
#   binary vectors):
#        marital_status | age | income | single | married | divorced
#            single     |  25 |  45000 |     1  |       0 |        0
#            married    |  28 |  52000 |     0  |       1 |        0
#           divorced    |  32 |  58000 |     1  |       0 |        1            !!
#            single     |  35 |  62000 |     1  |       0 |        0
#            married    |  40 |  68000 |     0  |       1 |        0
#            single     |  27 |  75000 |     1  |       0 |        0
#            married    |  30 |  81000 |     0  |       1 |        0
#           divorced    |  34 |  89000 |     1  |       0 |        1            !!
#            single     |  37 |  92000 |     1  |       0 |        0
#            married    |  41 |  97000 |     0  |       1 |        0
#           divorced    |  29 | 102000 |     1  |       0 |        1            !!
#            single     |  33 | 108000 |     1  |       0 |        0
#            married    |  38 | 115000 |     0  |       1 |        0

#   When this happens, at least two of the dummy variables will suffer from (perfect) multicollinearity. That is, 
#   they’ll be (perfectly) correlated. This problem of Dummy Variable Trap will cause incorrect calculations of 
#   the ML algorithm/model's parameters (i.e. coefficients/weights/biases) of the ML algorithms/models (Once you 
#   fully understand OHE ML Encoding Technique, you will understand how this will be a problem at a more
#   algorithmic/technical level!! I will not be explaning this too much in-depth).

#   Hence, the rule of thumb to avoid the Dummy Variable Trap is to drop/remove one of the dummy variables. This
#   will not affect the OHE ML Encoding Technique as it can be thought as there is no need to represent n number
#   of nominal categorical variables with n number of dummmy variables. Technically, you can represent n number 
#   of nominal categorical variables with n-1 number of dummmy variables, as in this example, if you represent
#   the 3 nominal categorical variables with just 2 nominal categorical variables like so:
#        marital_status | age | income | single | married
#            single     |  25 |  45000 |     1  |       0      
#            married    |  28 |  52000 |     0  |       1
#           divorced    |  32 |  58000 |     1  |       0 
#            single     |  35 |  62000 |     1  |       0
#            married    |  40 |  68000 |     0  |       1
#            single     |  27 |  75000 |     1  |       0
#            married    |  30 |  81000 |     0  |       1
#           divorced    |  34 |  89000 |     1  |       0
#            single     |  37 |  92000 |     1  |       0
#            married    |  41 |  97000 |     0  |       1
#           divorced    |  29 | 102000 |     1  |       0
#            single     |  33 | 108000 |     1  |       0
#            married    |  38 | 115000 |     0  |       1
#   
#   it can be thought as if a person/row with both the 'single' and 'married' dummy variable columns, storing binary 
#   value, 0 or 1 (aka binary vectors), have a value of 0, then the person/row must belong to the 'divorced' 
#   categorical variable.

#   By dropping/removing any one of the dummy variable columns, we prevented the problem of the Dummy Variable
#   Trap, as well as maintaining the functionality of the OHE ML Encoding Technique.

#   Source(s): https://www.statology.org/dummy-variable-trap/ (Statology)

# After dropping these 2 columns, this concatenated dummy variable columns, storing binary values, 0 or 1 (aka binary 
# vectors), and the initial dataset, is now a One-hot Encoded dataset
one_hot_encoded_dataset = concatenated_dummy_variable_columns_storing_binary_values_with_dataset.drop(['house_location', 'west windsor'], axis='columns')
print(one_hot_encoded_dataset)

# Output:
#     area   price  monroe township  robinsville
# 0   2600  550000                1            0
# 1   3000  565000                1            0
# 2   3200  610000                1            0
# 3   3600  680000                1            0
# 4   4000  725000                1            0
# 5   2600  575000                0            1
# 6   2900  600000                0            1
# 7   3100  620000                0            1
# 8   3600  695000                0            1
# 9   2600  585000                0            0
# 10  2800  615000                0            0
# 11  3300  650000                0            0
# 12  3600  710000                0            0

