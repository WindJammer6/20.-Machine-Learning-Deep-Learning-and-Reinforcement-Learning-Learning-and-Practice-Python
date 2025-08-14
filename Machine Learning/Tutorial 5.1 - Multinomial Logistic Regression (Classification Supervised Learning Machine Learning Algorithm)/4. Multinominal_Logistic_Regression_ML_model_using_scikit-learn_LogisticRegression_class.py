import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Reading the 'fruit_categories_with_multiple_independent_variables_and_multiple_nominal_categorical_
# dependent_variables.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("fruit_categories_with_multiple_independent_variables_and_multiple_nominal_categorical_dependent_variables.csv")
print(dataset)


# /////////////////////////////////////////////////////////////////////////////


# Since the 'Fruit Type' column in the dataset is a nominal (since it stores the categorical values, 'apple',
# 'banana' and 'orange', with no intrinsic order or ranking among them) non-numerical column 
# dependent categorical variable, hence, we need to first use the One-hot Encoding Machine Learning 
# Encoding Technique to convert it to a numerical column dependent categorical variable (refer to the
# 'Tutorial 4 - Machine Learning Encoding Techniques, One-hot Encoding Machine Learning Encoding 
# Technique and One-hot Encoding Machine Learning Encoding Technique (Machine Learning Technique)' folder
# for more information on the One-hot Encoding Machine Learning Encoding Technique)

# Manually creating an One-hot Encoded dataset of nominal categorical variables using custom mapping:
# Creating a mapping for the 'Fruit Type' column, which is storing the nominal categorical variables, for each 
# nominal categorical variables, 'apple', 'banana' and 'orange', to the values of 1, 2 and 3 respectively
custom_mapping = {
    "Apple": 1,
    "Banana": 2,
    "Orange": 3
}

# Applying the mapping for the 'Fruit Type' column, to the 'Fruit Type' column in the initial dataset
dataset['Fruit Type'] = dataset['Fruit Type'].map(custom_mapping)
print(dataset)


# ////////////////////////////////////////////////////////////////////////////////


# Using the Train Test Split Machine Learning (ML) Model Evaluation Technique to divide the dataset into two 
# seperate subsets: the training dataset subset and the test dataset subset (refer to the 'Tutorial 12 - Train 
# Test Split Machine Learning Model Evaluation Technique (Machine Learning Technique)' folder for more on the 
# Train Test Split Machine Learning (ML) Model Evaluation Technique and explanation on the code).

# Doing the Train Test Split Machine Learning (ML) Model Evaluation Technique using Scikit-learn's 'train_test_split' 
# Instance Method in the 'sklearn.model_selection' class
independent_variables = dataset[['Weight', 'Color', 'Size', 'Sweetness']]
dependent_variable = dataset['Fruit Type']
print(independent_variables)
print(dependent_variable)

independent_variables_training_dataset_subset, independent_variables_test_dataset_subset, dependent_variable_training_dataset_subset, dependent_variable_test_dataset_subset = train_test_split(independent_variables, dependent_variable, test_size=0.1, random_state=10)

print(f'The independent variables training dataset subset: \n{independent_variables_training_dataset_subset}')

print(f'The independent variables test dataset subset: \n{independent_variables_test_dataset_subset}')

print(f'The dependent variables training dataset subset: \n{dependent_variable_training_dataset_subset}')

print(f'The dependent variables test dataset subset: \n{dependent_variable_test_dataset_subset}')


# We will later continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to 
# evaluate the performance of the MNLR ML model once it is trained.


# ////////////////////////////////////////////////////////////////////////////////


# Using the Train Test Split Machine Learning (ML) Model Evaluation Technique to divide the dataset into two 
# seperate subsets: the training dataset subset and the test dataset subset (refer to the 'Tutorial 12 - Train 
# Test Split Machine Learning Model Evaluation Technique (Machine Learning Technique)' folder for more on the 
# Train Test Split Machine Learning (ML) Model Evaluation Technique and explanation on the code).

# Doing the Train Test Split Machine Learning (ML) Model Evaluation Technique using Scikit-learn's 'train_test_split' 
# Instance Method in the 'sklearn.model_selection' class
independent_variables = dataset[['Weight', 'Color', 'Size', 'Sweetness']]
dependent_variable = dataset['Fruit Type']
print(independent_variables)
print(dependent_variable)

independent_variables_training_dataset_subset, independent_variables_test_dataset_subset, dependent_variable_training_dataset_subset, dependent_variable_test_dataset_subset = train_test_split(independent_variables, dependent_variable, test_size=0.1, random_state=10)

print(f'The independent variables training dataset subset: \n{independent_variables_training_dataset_subset}')

print(f'The independent variables test dataset subset: \n{independent_variables_test_dataset_subset}')

print(f'The dependent variables training dataset subset: \n{dependent_variable_training_dataset_subset}')

print(f'The dependent variables test dataset subset: \n{dependent_variable_test_dataset_subset}')


# We will later continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to 
# evaluate the performance of the MNLR ML model once it is trained.


# //////////////////////////////////////////////////////////////////////////////


# Using the Scikit-learn Machine Learning (ML) Python library, we will now use its MNLR ML algorithm class (refer to the 
# '3. About_scikit-learn_Machine_Learning_Python_library.txt' file in the 'Tutorial 1 - What is Machine Learning' folder),
#       'LogisticRegression()' from the 'sklearn.linear_model' Scikit-learn ML Python library's submodule 

# to create a 'MNLR ML model' class object/instance, and allowing us to access the 
# MNLR ML algorithm class's Instance Methods such as:
#    - '.fit()'           - this Instance Method trains the MNLR ML algorithm to become a MNLR ML model
#    - '.predict()'       - this Instance Method makes a prediction using the MNLR ML model
#    - '.predict_proba()' - this Instance Method returns the predicted probability of the prediction 
#                           being in each category of the categorical dependent variable
#    - '.score()'         - this Instance Method evaluates the accuracy of the MNLR ML model

# and the MNLR ML algorithm class's attributes such as (note that the trailing underscore ('_') is a convention 
# used in the Scikit-learn Machine Learning (ML) Python library to represent attributes):
#    - '.coef_'      - this attribute represents the values of the weights/gradients/coefficients of the independent 
#                      variables/features 'x1', 'x2', 'x3' ... 'x?', 'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)' 
#                      respectively, in the '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi', 
#                      for 𝑖 = 1, 2, … , 𝐾  in the mathematical equation representing the MNLR ML algorithm, 
#                      'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)' 
#    - '.intercept_' - this attribute represents the value of the bias/intercept 'bi', in the 
#                      '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi' for 𝑖 = 1, 2, … , 𝐾  
#                      in the mathematical equation representing the MNLR ML algorithm, 
#                      'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)'


# Creating a 'MNLR ML model' class object/instance
multinomial_logistic_regression_ML_model = LogisticRegression()


# Training the 'MNLR ML model' class object/instance

# For the '.fit(X, y, sample_weight=None)' Instance Method of the 'MNLR ML model' class, 
# it takes 3 parameters, with the first 2 parameters are the more important ones:
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MNLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MNLR ML algorithm
print(multinomial_logistic_regression_ML_model.fit(independent_variables_training_dataset_subset, dependent_variable_training_dataset_subset))


# The '.coef_' attribute of the 'MNLR ML model' class shows the values of the weights/gradients/coefficients of the 
# independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)' 
# respectively, in the '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi', 
# for 𝑖 = 1, 2, … , 𝐾  in the mathematical equation representing the MNLR ML algorithm, 
# 'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)'

# (Note: The multiple coefficient values of the independent variables/features, 
#        'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)', will be displayed in the output in order, in a Python list, 
#        for example, '(refer to the output of the '.coef_' attribute of the 'MNLR ML model' class object/instance below)', 
#        where the list in the first index '[-0.10179905  1.21821349  1.45466895 -0.81593939]' will be for the 
#        weights/gradients/coefficients for 𝑖 = 1, '𝑧1 = m(z1, 1) * x1 + m(z1, 2) * x2 + m(z1, 3) * x3 + ... + m(z1, ?) * x? + b1', 
#        with '-0.10179905' will be 'm(z1, 1)', '1.21821349' will be ' m(z1, 2)', '1.45466895' will be ' m(z1, 3)', and
#        '-0.81593939' will be ' m(z1, 4)', then the list in the second index '[-0.06816823 -0.50291189  0.19141042  1.47570913]'
#        will be for the weights/gradients/coefficients for 𝑖 = 2, '𝑧2 = m(z2, 1) * x2 + m(z2, 2) * x2 + m(z2, 3) * x3 + ... + 
#        m(z2, ?) * x? + b1', etc.)
print(multinomial_logistic_regression_ML_model.coef_)                        # output: [[-0.10179905  1.21821349  1.45466895 -0.81593939]
                                                                             #          [-0.06816823 -0.50291189  0.19141042  1.47570913]
                                                                             #          [ 0.16996728 -0.7153016  -1.64607937 -0.65976974]]


# The '.intercept_' attribute of the 'MNLR ML model' class shows the value of the bias/intercept, 'bi', in the 
# '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi' for 𝑖 = 1, 2, … , 𝐾  in the 
# mathematical equation representing the MNLR ML algorithm, 'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)'
print(multinomial_logistic_regression_ML_model.intercept_)                   # output: [ 2.30938371  1.15525481 -3.46463852]



# Making predictions with the 'MNLR ML model' class object/instance

# For the '.predict(X)' Instance Method of the 'MNLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'MNLR ML model' class object/instance to make a prediction of. The output of this Instance Method is the predicted
#      nominal categorical dependent variable, 'y', of the independent variables/features by the 'MNLR ML model' class 
#      object/instance

# Hence, in this case, 
#   -> the output of this '.predict()' Instance Method of the 'MNLR ML model' class is the (nominal categorical) dependent
#      variable, 'y' '[1 3 2 3 2 2 1 2 3]', using the mathematical equation representing the MNLR ML algorithm, 
#      'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)', where '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi', 
#      for 𝑖 = 1, 2, … , 𝐾, and finding the all the different real-number-of-z-now-a-value-between-0-and-1, 'softmax(zi)',
#      probabilities, for 𝑖 = 1, 2, … , 𝐾, and since 'y' is the (multiple nominal categorical) dependent variable, the particular 
#      nominal category 'zi' function that has the largest 'softmax(zi)', probability for a set of values of the independent 
#      variables/features for the MNLR ML algorithm will be interpreted as the (predicted) 'y' (multiple nominal categorical) 
#      dependent variable value to be the category represented by that value of 𝑖.

#   -> the input 'X' parameter of this '.predict()' Instance Method of the 'MNLR ML model' class is the independent 
#      variables/features, 'x1', 'x2', 'x3', 'x4' in the '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi', 
#      for 𝑖 = 1, 2, … , 𝐾  in the mathematical equation representing the MNLR ML algorithm, 
#      'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)'

# Note: the multiple independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the '.predict(X)' 
# Instance Method of the 'MNLR ML model' class will need to be in order, in a 2D array
print(multinomial_logistic_regression_ML_model.predict(independent_variables_test_dataset_subset))            # output: [1 3 2 3 2 2 1 2 3]

# These are the sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
# '.predict(X)' Instance Method of the 'MNLR ML model' class that we are using to make the respective predictions using the 
# MNLR ML model
print(independent_variables_test_dataset_subset)
# Output:
#     Weight  Color  Size  Sweetness
# 3      140      6     8          5
# 19     195      8    10          6
# 32     102      7     5         10
# 41     217      8     9          8
# 46     127      5     7          8
# 34     122      5     7          8
# 48     153      9     8          7
# 47     137      4     8          7
# 52     203      9    10          9


# Interpreting the predicted 'y' (nominal categorical) dependent variable values using the MNLR ML model for this dataset:
# Just looking at the first set of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters 
# in the '.predict(X)' Instance Method of the 'MNLR ML model' class in the 'independent_variables_test_dataset_subset' 
# variable, of 'Weight = 140', 'Color = 6', 'Size = 8', and 'Sweetness = 5' and since for this 'fruit_categories_with_
# multiple_independent_variables_and_multiple_nominal_categorical_dependent_variables.csv' dataset a 'y' (nominal 
# categorical) dependent variable value of 1 (for i = 1) represents the category 'apple', a 'y' (nominal categorical) 
# dependent variable value of 2 (for i = 2) represents the category 'banana', while a 'y' (nominal categorical) 
# dependent variable value of 3 (for i = 3) represents the category 'orange'

# hence, the respective predicted 'Fruit Type' ('y') (nominal categorical) dependent variable value using the MNLR ML 
# model is:
# - '1', representing the 'apple' category for 'Weight = 140', 'Color = 6', 'Size = 8', and 'Sweetness = 5'



# The predicted probability of the predictions with the 'MNLR ML model' class object/instance being in each category of 
# the categorical dependent variable
# For the '.predict_proba(X)' Instance Method of the 'MNLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'MNLR ML model' class object/instance to return the predicted probability of the prediction with the 
#      'MNLR ML model' class object/instance being in each category of the categorical dependent variable. The output 
#      of this Instance Method is the predicted probability of the prediction with the 'MNLR ML model' class 
#      object/instance being in each category of the categorical dependent variable, returned as a 2D array

# Note: The categorical dependent variable labels each category with a value in the dataset, for example, in this 
#       'fruit_categories_with_multiple_independent_variables_and_multiple_nominal_categorical_dependent_
#       variables.csv' dataset, the 'apple' category is represented by the value of 1 (for i = 1), the 'banana' category is 
#       represented by the value of 2 (for i = 2), while the the 'banana' category is represented by the value of 3 (for i = 3)

#       Hence, the returned 2D array of the predicted probability of the prediction with the 'MNLR ML model' class 
#       object/instance being in each category of the categorical dependent variable will order the category of the 
#       categorical dependent variable represented by the smallest value as the first index of the 1D arrays in the
#       returned 2D array, and the category of the categorical dependent variable represented by the second smallest 
#       value as the second index of the 1D arrays in the returned 2D array, the category of the categorical dependent 
#       variable represented by the represented by the third smallest value as the third index of the 1D arrays in 
#       the returned 2D array, and so on...

#       For example, for the first set of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input 
#       parameters in the '.predict(X)' Instance Method of the 'MNLR ML model' class in the 'independent_variables_test_
#       dataset_subset' variable by the 'MNLR ML model' class object/instance of the 'fruit_categories_with_multiple_
#       independent_variables_and_multiple_nominal_categorical_dependent_variables.csv' dataset, from the output of this 
#       '.predict_proba()' Instance Method, 
#           [[9.62088059e-01 4.25763524e-03 3.36543058e-02]
#            ...]

#       since the 'apple' category of the categorical dependent variable represented by the smallest 
#       value (of 1) (for i = 1) is the first index of the 1D arrays in the returned 2D array, the 'banana' category 
#       of the categorical dependent variable represented by the second smallest value (of 2) (for i = 2) as the second 
#       index of the 1D arrays in the returned 2D array, and the 'orange' category of the categorical dependent variable 
#       represented by the third smallest value (of 3) (for i = 3) is the third index of the 1D arrays in the returned 
#       2D array, hence, the respective predicted probability of the predictions of the first set of (multiple) independent 
#       variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the '.predict(X)' Instance Method of the 
#       'MNLR ML model' class in the 'independent_variables_test_dataset_subset' variable, of 'Weight = 140', 
#       'Color = 6', 'Size = 8', and 'Sweetness = 5' is:
#       - for 'Weight = 140', 'Color = 6', 'Size = 8', and 'Sweetness = 5', there is a predicted probability of 
#         0.962088059, for the first index of the 1D arrays in the returned 2D array, which represents the 'apple' 
#         category, which is the categorical dependent variable with the smallest value of 1 (for i = 1), a predicted probability 
#         of 0.00425763524, for the second index of the 1D arrays in the returned 2D array, which represents 
#         'banana' category, which is the categorical dependent variable with the second smallest value of 2 (for i = 2), and a 
#         predicted probability of 0.0336543058, for the third index of the 1D arrays in the returned 2D array, 
#         which represents 'orange' category, which is the categorical dependent variable with the third smallest 
#         value of 3 (for i = 3) 
# 
#       And since, recalling that the since the MNLR ML algorithm determines the (predicted) 'y' (nominal 
#       categorical) dependent variable value for a set of values of the independent variables/features (see the section 
#       'How does the Multinomial Logistic Regression (MNLR) Machine Learning (ML) algorithm work' in the 
#       '1. What_is_Multinomial_Logistic_Regression.txt' file) using the particular nominal category 'zi' function 
#       that has the largest real-number-of-z-now-a-value-between-0-and-1, 'softmax(zi)', probability, 
#       - for 'Weight = 140', 'Color = 6', 'Size = 8', the predicted probability is 0.962088059 for the 'apple' category 
#         represented by a value of 1 (for i = 1), a predicted probability of 0.00425763524 for the 'banana' category 
#         represented by a value of 2 (for i = 2), and a predicted probability of 0.0336543058 for the 'banana' category 
#         represented by a value of 3 (for i = 3), the (predicted) 'y' (nominal categorical) dependent variable 
#         value will be 1 due to the 'apple' category represented by a value of 1 (for i = 1) having the largest 
#         predicted probability

#       (Note: You can see that this aligns with the (predicted) 'y' (nominal categorical) dependent variable value for 
#        the first set of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
#        '.predict(X)' Instance Method of the 'MNLR ML model' class in the 'independent_variables_test_dataset_subset'
#        of '[1 3 2 3 2 2 1 2 3]')
print(multinomial_logistic_regression_ML_model.predict_proba(independent_variables_test_dataset_subset))
# Output:
# [[9.62088059e-01 4.25763524e-03 3.36543058e-02]
#  [1.56836325e-01 1.11624630e-04 8.43052051e-01]
#  [1.08127360e-03 9.98914440e-01 4.28614405e-06]
#  [1.55135559e-05 8.00712303e-06 9.99976479e-01]
#  [1.78750690e-02 9.77642112e-01 4.48281886e-03]
#  [2.11450910e-02 9.77492273e-01 1.36263624e-03]
#  [9.91289542e-01 3.80279416e-03 4.90766359e-03]
#  [7.01965746e-02 8.58822936e-01 7.09804894e-02]
#  [8.27465405e-02 1.33408749e-02 9.03912585e-01]]


# //////////////////////////////////////////////////////////////////////////


# Continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to evaluate the 
# performance of the trained MNLR ML model (refer to the 'Tutorial 12 - Train Test Split Machine Learning Model 
# Evaluation Technique (Machine Learning Technique)' folder for more on the Train Test Split Machine Learning 
# (ML) Model Evaluation Technique and explanation on the code).

# Evaluating the trained MNLR ML model using the '.score()' Instance Method of the 'Multinomial Logistic Regression 
# (MNLR) ML model' class, using the test dataset subset obtained from the Train Test Split ML Model Evaluation Technique

# Apart from the the MNLR ML algorithm class giving us access to the '.fit()' and '.predict()' Instance Methods and the
# '.coef_' and '.intercept_' attributes, the MNLR ML algorithm class also gives us access to the '.score()' Instance
# Method, which allows us to calculate how accurate the MNLR ML model is:

# How the MNLR ML algorithm class's '.score(X, y)' Instance method calculate how accurate our MNLR ML model is,
# is that it takes in 2 parameters, 
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the MNLR ML algorithm (In this tutorial, this will be the 'independent_variables_test_dataset_subset'
#      obtained from the Train Test Split ML Model Evaluation Technique)
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      MNLR ML algorithm (In this tutorial, this will be the 'dependent_variables_test_dataset_subset'
#      obtained from the Train Test Split ML Model Evaluation Technique)

# It then uses the MNLR ML model, which is trained with the 'independent_variables_training_dataset_subset', to 
# predict each of the predicted 'Fruit Type' nominal categorical dependent variable for each of the independent 
# variables/features in the 'independent_variables_test_dataset_subset', and compare them with the provided actual 
# predicted 'Fruit Type' nominal categorical dependent variable in the 'dependent_variable_test_dataset_subset'. 
# It then uses some mathematical function (not super important honestly) to calculate the percentage (%) closeness 
# of the predicted predicted 'Fruit Type' nominal categorical dependent variable by the MNLR ML model, which is 
# trained with the 'independent_variables_training_dataset_subset', with the provided actual predicted 'Fruit Type' 
# nominal categorical dependent variable in the 'dependent_variable_test_dataset_subset', as an indicator score 
# for how accurate the MNLR ML model is 
print(multinomial_logistic_regression_ML_model.score(independent_variables_test_dataset_subset, dependent_variable_test_dataset_subset))
# Output: 1.0

print(f'     Fruit Type\n{dependent_variable_test_dataset_subset}')
# Output:
#      Fruit Type
# 3     1
# 19    3
# 32    2
# 41    3
# 46    2
# 34    2
# 48    1
# 47    2
# 52    3

# Interpreting the evaluation accuracy score of the MNLR ML model for this dataset:
# The evaluation accuracy score of the MNLR ML model is 1.0 for this dataset, meaning that our MNLR ML model is interpreted
# as perfect. But this only happens since our dataset size is small, and is usually not the case.

# Our MNLR ML model is interpreted as perfect because the predicted 'y' (nominal categorical) dependent variable value for
# the independent variables/features in the 'independent_variables_training_dataset_subset' of '[1 3 2 3 2 2 1 2 3]' (see 
# the output of the '.predict()' Instance Method above) and the provided actual predicted 'Fruit Type' (nominal categorical 
# dependent variable) in the 'dependent_variable_test_dataset_subset' of '[1 3 2 3 2 2 1 2 3]' are both exactly the same.

# But if we use larger datasets, the MNLR ML model is bound to make errors in its predicted 'y' (nominal categorical) 
# dependent variable value for the independent variables/features that will not be the same as the provided actual 
# predicted nominal categorical dependent variable, and will cause the evaluation accuracy score to be less than 1.0.


# //////////////////////////////////////////////////////////////////////////


# Some key observations from all this:

# So how is this predicted value/output made by the 'Multinomial Logistic Regression (MNLR) ML model' class 
# object/instance?
# Essentially, looking at in the mathematical equation representing the MNLR ML algorithm, 
# 'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)', where 
# '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi', for 𝑖 = 1, 2, … , 𝐾, since:
# - the '.coef_' attribute of the 'MNLR ML model' class shows the values of the weights/gradients/coefficients of the 
#   independent variables/features 'x1', 'x2', 'x3' and 'x4', 'm(z1, 1)' ('-0.10179905'), 'm(z1, 2)' ('1.21821349'), 
#   'm(z1, 3)' ('1.45466895'), 'm(z1, 4)' ('-0.81593939'), m(z2, 1)' ('-0.06816823'), 'm(z2, 2)' ('-0.50291189'), etc.  
# - the input 'X' parameter of this '.predict()' Instance Method of the 'MNLR ML model' class is the independent
#   variables/features, ('Weight') 'x1' ('140'), ('Color') 'x2' ('6'), ('Size') 'x3' ('8'), ('Sweetness') 'x4' ('8')
# - the '.intercept_' attribute of the 'MNLR ML model' class shows the values of the bias/intercepts 'b1' ('2.30938371'), 
#   'b2' ('1.15525481'), 'b3 ('-3.46463852')
# - '𝑒' is the Euler's number ~ 2.71828


# (Recall that, since in this context the 'apple' category is represented by a value of 1 (for i = 1), the 'banana' 
#  category represented by a value of 2 (for i = 2), the 'orange' category represented by a value of 3 (for i = 3),
#  and hence for 𝑖 = 1, 2, … , 𝐾, with 𝐾 representing the total number of nominal categories (in this context there 
#  is 3 nominal categories))

# z1 =  'm(z1, 1)'  * 'x1' +  'm(z1, 2)'   *  'x2' +   'm(z1, 3)'   * 'x3'  +   'm(z1, 4)'  *  'x4' + ... +    'bz1'
z1   = -0.10179905  * 140  +  1.21821349   *    6  +   1.45466895   *  8    + (-0.81593939) *    5        + 2.30938371         # = 2.9244522999999982
print(z1)

# z2 =  'm(z2, 1)'  * 'x1' +  'm(z2, 2)'   *  'x2' +   'm(z2, 3)'   * 'x3'  +   'm(z2, 4)'  *  'x4' + ... +    'bz2'
z2   =(-0.06816823) * 140  + (-0.50291189) *    6  +  0.19141042   *  8     +   1.47570913  *    5         + 1.15525481        # = -2.495939719999999
print(z2)

# z3 =  'm(z3, 1)'  * 'x1' +  'm(z3, 2)'   *  'x2' +  'm(z3, 3)'    * 'x3'  +   'm(z3, 4)'  *  'x4' + ... +    'bz2'
z3   = 0.16996728   * 140  + (-0.7153016)  *    6  + (-1.64607937)  *  8    + (-0.65976974) *    5        + (-3.46463852)      # = -0.42851257999999914 
print(z3)


# When 𝑖 = 0, the 'apple' category
#               𝑒     ^  'z0' /       𝑒     ^ 'z0' +       𝑒     ^  'z1'   +     𝑒     ^  'z2'         = 'softmax(z0)'
softmaxz1 = (2.71828 ** (z1))  / ((2.71828 ** (z1))  + (2.71828 **  (z2))  + (2.71828 **  (z3)))     # = 0.9620879675714622
print(softmaxz1)

# When 𝑖 = 1, the 'banana' category
#               𝑒     ^  'z1' /       𝑒     ^ 'z0' +       𝑒     ^  'z1'   +     𝑒     ^  'z2'         = 'softmax(z1)'
softmaxz2 = (2.71828 ** (z2))  / ((2.71828 ** (z1)) + (2.71828 ** (z2))  + (2.71828 **  (z3)))       # = 0.004257650638778712
print(softmaxz2)

# When 𝑖 = 2, 'orange' category
#               𝑒     ^  'z2' /       𝑒     ^ 'z0' +       𝑒     ^  'z1'   +     𝑒     ^  'z2'         = 'softmax(z2)'
softmaxz3 = (2.71828 ** (z3))  / ((2.71828 ** (z1)) + (2.71828 ** (z2))  + (2.71828 **  (z3)))       # = 0.033654381789759
print(softmaxz3)


# Showing that the sum of all the real-number-of-z-now-a-value-between-0-and-1, 'softmax(zi)',
# probabilities is equal to 1
print(softmaxz1 + softmaxz2 + softmaxz3)        # = ~ 1.0


# And since the output of the softmax function is the mapped real number of zi to a value between 0 and 1 
# is so that the real-number-of-z-now-a-value-between-0-and-1, 'softmax(zi)', can be interpreted as a probability!

# Recall that the since the MNLR ML algorithm determines the (predicted) 'y' (nominal categorical) dependent 
# variable value for a set of values of the independent variables/features (see the section 'How does the Multinomial 
# Logistic Regression (MNLR) Machine Learning (ML) algorithm work' in the '1. What_is_Multinomial_Logistic_
# Regression.txt' file) using the particular nominal category 'zi' function that has the largest 
# real-number-of-z-now-a-value-between-0-and-1, 'softmax(zi)', probability. 

# Hence, since the real-number-of-z-now-a-value-between-0-and-1, 'softmax(zi)', probability for the 'apple' category 
# represented by a value of 1 (for i = 1) is the largest, the (predicted) 'y' (nominal categorical) dependent 
# variable value for a set of values of the independent variables/features for the MNLR ML algorithm will be 
# 1.