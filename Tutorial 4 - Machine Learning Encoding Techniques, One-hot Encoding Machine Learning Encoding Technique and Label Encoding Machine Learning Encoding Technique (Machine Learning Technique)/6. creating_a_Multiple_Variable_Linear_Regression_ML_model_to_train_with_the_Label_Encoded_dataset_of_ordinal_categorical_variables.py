import pandas as pd
from sklearn.linear_model import LinearRegression

# Manually creating a Label Encoded dataset of ordinal categorical variables

# Reading the 'salary_with_ordinal_categorical_variables.csv' dataset from the CSV file
dataset = pd.read_csv("salary_with_ordinal_categorical_variables.csv")
print(dataset)


# Creating a mapping for the 'education_level' column, which is storing the ordinal categorical variables, for each 
# ordinal categorical variables, 'graduate', 'master's' and 'PhD', to the values of 0, 1 and 2 respectively, which 
# indicates/preserves the specific, inherent order or ranking among them (since for Education level: 
# PhD > master's > graduate).
# (see the next file '7. creating_a_Label_Encoded_dataset_of_ordinal_categorical_variables_using_scikit-learn_
#  LabelEncoder_class.py' for why I decided to rename the ordinal categorical variables to 'a_graduate', 'b_master's'
#  and 'c_PhD')
custom_mapping = {
    "a_graduate": 0,
    "b_master's": 1,
    "c_PhD": 2
}


# Applying the mapping for the 'education_level' column, to the 'education_level' column in the initial dataset
dataset['education_level'] = dataset['education_level'].map(custom_mapping)


# After applying the mapping for the 'education_level' column, to the 'education_level' column in the initial 
# dataset, this initial dataset is now a Label Encoded dataset
label_encoded_dataset = dataset 
print(label_encoded_dataset)


# ////////////////////////////////////////////////////////////////////////////////


# Creating the Multiple Variable Linear Regression (MVLR) ML model that we will be testing our 
# Label Encoded dataset on (refer to the 'Tutorial 2.1 - Multiple Variable Linear Regression (Supervised 
# Regression Machine Learning Algorithm)' for more about the MVLR ML algorithm)

# Creating a 'MVLR ML model' class object/instance
multiple_variable_linear_regression_ML_model = LinearRegression()



# Training the 'MVLR ML model' class object/instance with the Label Encoded dataset

# For the '.fit(X, y, sample_weight=None)' Instance Method of the 'MVLR ML model' class, 
# it takes 3 parameters, with the first 2 parameters are the more important ones:
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MVLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MVLR ML algorithm

# Getting the independent variables/features, which is all the columns in the 'label_encoded_dataset' except the 
# 'salary' column (the dependent variable). Hence, here we dropped/removed the 'salary' column, and taking the  
# remaining 'label_encoded_dataset' as the independent variables/features

# The dependent variable is the 'salary' column in the initial dataset

# (Note: What is 'one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']]'?
#        From the '3. Single_Variable_Linear_Regression_ML model_using_scikit-learn_LinearRegression_class.py' file in 
#        the 'Tutorial 2 - Linear Regression and Single Variable Linear Regression (Regression Supervised Learning 
#        Machine Learning Algorithm)' folder, we know that 'dataset[['area']]' returns a DataFrame, essentially a 2D 
#        array. Hence, 'one_hot_encoded_dataset[['area', 'monroe township', 'robinsville']]' also returns a 2D array, 
#        but while 'dataset[['area']]' returns a 2D array DataFrame with 1 column, 'one_hot_encoded_dataset[['area', 
#        'monroe township', 'robinsville']]' returns a 2D array DataFrame with 3 columns.

#        Also, the order of the columns/independent variables/features 'education_level', 'age' in 
#        the code 'label_encoded_dataset[['education_level', 'age']]' matters, as in the mathematical 
#        equation representing the Multiple Variable Linear Regression (MVLR) ML algorithm, 'y = m1*x1 + m2*x2 + ... 
#        + m?*x? + b', the first column name in the code 'label_encoded_dataset[['education_level', 'age']]' 
#        will be 'x1', the second column name will be 'x2', and so on... In the mathematical equation for the MVLR ML 
#        algorithm in this context, aka the Machine Learning (ML) model for this context, it is:
#                    'y'       'x1'        'x2'       
#                   salary = m1*age + m2*monroe township + b

#        (This only applies for the MVLR ML algorithm, and not for the Single Variable Linear Regression (SVLR) ML algorithm,
#        since the SVLR ML algorithm only has 1 column/independent variable/feature):
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

# (Note: The multiple coefficient values of the independent variables/features, 'm1', 'm2', ... 'm?', will be displayed
#        in the output in order, in a Python list, for example, '[3534.27035965  875.92275438]', where 
#        '3534.27035965' will be 'm1' and '875.92275438' will be 'm2')
print(multiple_variable_linear_regression_ML_model.coef_)                          # output: [3534.27035965  875.92275438]

# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the intercept (NOT y-intercept, refer to the
# '1. What_is_Multiple_Variable_Linear_Regression.txt' file in the 'Tutorial 2.1 - Multiple Variable Linear Regression 
# (Supervised Regression Machine Learning Algorithm)' folder), 'b', in the mathematical equation representing the MVLR ML 
# algorithm, 'y = m1*x1 + m2*x2 + m3*x3 + ... + m?*x? + b'  
print(multiple_variable_linear_regression_ML_model.intercept_)                     # output: 33266.828087167065



# Making predictions with the 'MVLR ML model' class object/instance, which is trained with the Label Encoded dataset
# For the '.predict(X)' Instance Method of the 'MVLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'MVLR ML ML model' class object/instance to make a prediction of. The output of this Instance Method is the predicted
#      dependent variable, 'y', of the test independent variables/features by the 'MVLR ML ML model' class object/instance

# Hence, in this case, 
#   -> the output of this '.predict()' Instance Method of the 'MVLR ML model' class is the dependent variable, 'y' 
#      ('59544.51071871'), in the mathematical equation representing the MVLR ML algorithm, 
#      'y = m1*x1 + m2*x2 + ... + m?*x? + b'  
#   -> the input 'X' parameter of this '.predict()' Instance Method of the 'MVLR ML model' class is the independent 
#      variables/features, 'x1' ('0'), 'x2' ('30')] in the mathematical equation representing the MVLR ML 
#      algorithm, 'y = m1*x1 + m2*x2 + ... + m?*x? + b'  

# Note: the multiple independent variables/features, 'x1', 'x2', ... 'x?', input parameters in the '.predict(X)' 
# Instance Method of the 'MVLR ML model' class will need to be in order, in a 2D array, where '0' will be 'x1', and 
# '30' will be 'x2', and that they cannot be in a list/1D array like, '[0,30]', but must be in a 2D array like, '[[0,30]]'


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

# How the MVLR ML algorithm class's '.score(X, y)' Instance method calculate how accurate our MVLR ML model is,
# is that it takes in 2 parameters, 
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MVLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MVLR ML algorithm

# It then uses the MVLR ML model, which is trained with the Label Encoded dataset, to re-predict each of the 'salary'
# (dependent variable) for each of the independent variables/features in the Label Encoded dataset, and compare them
# with the actual 'salary' (dependent variable) provided in the Label Encoded dataset. It then uses some mathematical
# function (not super important honestly) to calculate the percentage (%) closeness of the re-predicted 'salary' (dependent 
# variable) by the MVLR ML model, which is trained with the Label Encoded dataset, with the actual 'salary' (dependent 
# variable) provided in the Label Encoded dataset, as an indicator score for how accurate the MVLR ML model is 
print(multiple_variable_linear_regression_ML_model.score(independent_variables_or_features, dataset['salary']))

# Output: 0.9742667224853172
# This means that the MVLR ML model is 97% accurate 


