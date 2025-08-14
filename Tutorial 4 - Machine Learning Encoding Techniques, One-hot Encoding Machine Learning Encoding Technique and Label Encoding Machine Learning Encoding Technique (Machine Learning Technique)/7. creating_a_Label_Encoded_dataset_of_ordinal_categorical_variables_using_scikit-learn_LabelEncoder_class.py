import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Creating a Label Encoded dataset of ordinal categorical variables using Scikit-learn's 'LabelEncoder' class
# (Note: The  Scikit-learn's 'LabelEncoder' class does the exact same functionality as the code in the file 
# '5. manually_creating_a_Label_Encoded_dataset_of_ordinal_categorical_variables.py')

# Reading the 'salary_with_ordinal_categorical_variables.csv' dataset from the CSV file
dataset = pd.read_csv("salary_with_ordinal_categorical_variables.csv")
print(dataset)


# Creating the Label Encoded dataset of ordinal categorical variables by first creating a Scikit-learn's 
# 'LabelEncoder' class instance
labelencoder = LabelEncoder()



# Using the '.fit_transform()' Instance method of the Scikit-learn's 'LabelEncoder' class to 'fit' the ordinal
# categorical variables, in the 'education_level' column, into the 'LabelEncoder' to do Label Encoding on the 
# ordinal categorical variables. The Scikit-learn's 'LabelEncoder' class to 'fits' the ordinal categorical 
# variables in ASCII order.

# The 'fitted' ordinal categorical variables, in the 'education_level' column, into the 'LabelEncoder' is then
# assigned back into the initial dataset


# Why did I rename the ordinal categorical variables to 'a_graduate', 'b_master's' and 'c_PhD' in the 
# 'education_level' column?
# This is because the '.fit_transform()' Instance method of the Scikit-learn's 'LabelEncoder' class organises
# the ordinal categorical variables in ASCII order! (i.e. If you 'fit' the ordinal categorical variables
# 'graduate', 'master's' and 'PhD', into the 'LabelEncoder', it will assign the value of 0 to 'PhD', value of 
# 1 to 'graduate' and value of 2 to 'master's', because the ASCII order of the first letter of each ordinal 
# categorical variables is as follows: 'P' < 'g' < 'm')

# Hence, in order to indicate/preserve the specific, inherent order or ranking among the ordinal categorical 
# variables, 'graduate', 'master's' and 'PhD', by assigning the values of 0, 1 and 2 respectively (since for 
# Education level: PhD > master's > graduate) when using the  '.fit_transform()' Instance method of the 
# Scikit-learn's 'LabelEncoder' class, I needed to rename the ordinal categorical variables to 'a_graduate', 
# 'b_master's' and 'c_PhD', in the 'education_level' column, in order to set the ASCII order of the first 
# letter of each ordinal categorical variable to be the same ('a' < 'b' < 'c') as the specific, inherent order 
# or ranking among them ('a_graduate' < 'b_master's' < 'c_PhD')
dataset['education_level'] = labelencoder.fit_transform(dataset['education_level'])



# After assigning the 'fitted' ordinal categorical variables, in the 'education_level' column, into the 
# 'LabelEncoder', back into the initial dataset, this initial dataset is now a Label Encoded dataset
label_encoded_dataset = dataset 
print(label_encoded_dataset)


# ////////////////////////////////////////////////////////////////////////////////


# Creating the Multiple Variable Linear Regression (MVLR) ML model that we will be testing our 
# Label Encoded dataset on (refer to the 'Tutorial 2.1 - Multiple Variable Linear Regression (Supervised 
# Regression Machine Learning Algorithm)' for more about the MVLR ML algorithm)

# Creating a 'MVLR ML model' class object/instance
multiple_variable_linear_regression_ML_model = LinearRegression()


# Training the 'MVLR ML model' class object/instance with the Label Encoded dataset

# Getting the independent variables/features, which is all the columns in the 'label_encoded_dataset' except the 
# 'salary' column (the dependent variable). Hence, here we dropped/removed the 'salary' column, and taking the  
# remaining 'label_encoded_dataset' as the independent variables/features

# The dependent variable is the 'salary' column in the initial dataset
independent_variables_or_features = label_encoded_dataset.drop('salary', axis='columns')
print(independent_variables_or_features)
multiple_variable_linear_regression_ML_model.fit(independent_variables_or_features, dataset['salary'])
# multiple_variable_linear_regression_ML_model.fit(label_encoded_dataset[['education_level', 'age']], dataset['salary']) works too


# In this tutorial, the trained MVLR ML model's coefficients and intercept/biases might look pretty 
# cryptic due to the OHE ML Encoding Technique, but it works as ML parameters 
# (i.e. coefficients/weights/biases) for the trained MVLR ML model to give accurate predictions

# The '.coef_' attribute of the 'MVLR ML model' class shows the values of the 
# gradient/coefficient of the independent variables/features 'x1', 'x2', ... 'x?', 'm1', 'm2', ... 'm?' 
# respectively, in the mathematical equation representing the MVLR ML algorithm, 
# 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [3534.27035965  875.92275438]

# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file in the 'Tutorial 2.1 - Multiple Variable Linear Regression 
# (Supervised Regression Machine Learning Algorithm)' folder), 'b', in the mathematical equation representing the MVLR ML 
# algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.intercept_)                     # output: 33266.828087167065



# Making predictions with the 'MVLR ML model' class object/instance, which is trained with the Label Encoded dataset

# In this context, 'x1' represents the 'education_level' independent variable/feature and 'x2' represents the 'age' 
# independent variable/feature

# To get a prediction of a 'salary' (dependent variable) for a person of 'education_level' of 'a_graduate', the value of 'x1' 
# representing 'education_level' independent variable/feature) must be '0'
print(multiple_variable_linear_regression_ML_model.predict([[0,30]]))          # Output: 59544.51071871
# To get a prediction of a 'salary' (dependent variable) for a person of 'education_level' of 'b_master's', the value of 'x1' 
# (representing 'education_level' independent variable/feature) must be '1'
print(multiple_variable_linear_regression_ML_model.predict([[1, 26]]))         # Output: 59575.09006083
# To get a prediction of a 'salary' (dependent variable) for a person of 'education_level' of 'c_PhD', the value of 'x1' 
# (representing 'education_level' independent variable/feature) must be '2'
print(multiple_variable_linear_regression_ML_model.predict([[2,50]]))          # Output: 84131.50652572



# Apart from the the MVLR ML algorithm class giving us access to the '.fit()' and '.predict()' Instance Methods and the
# '.coef_' and '.intercept_' attributes, the MVLR ML algorithm class also gives us access to the '.score()' Instance
# Method, which allows us to calculate how accurate the MVLR ML model is:
print(multiple_variable_linear_regression_ML_model.score(independent_variables_or_features, dataset['salary']))

# Output: 0.9742667224853172
# This means that the MVLR ML model is 97% accurate 