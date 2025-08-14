import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Creating the visualisation of the dataset used for the Binary Logistic Regression (BLR) ML model on a graph:

# Reading the 'bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_
# dependent_variable.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_dependent_variable.csv")
print(dataset)

# Plotting the (scatter) graph using the Matplotlib library
plt.xlabel('age')
plt.ylabel('bought insurance or not bought insurance')
plt.scatter(dataset['age'], dataset['bought_insurance'], color='red', marker='+')


# ///////////////////////////////////////////////////////////////////////////////////////


# Using the Train Test Split Machine Learning (ML) Model Evaluation Technique to divide the dataset into two 
# seperate subsets: the training dataset subset and the test dataset subset (refer to the 'Tutorial 12 - Train 
# Test Split Machine Learning Model Evaluation Technique (Machine Learning Technique)' folder for more on the 
# Train Test Split Machine Learning (ML) Model Evaluation Technique and explanation on the code).

# Doing the Train Test Split Machine Learning (ML) Model Evaluation Technique using Scikit-learn's 'train_test_split' 
# Instance Method in the 'sklearn.model_selection' class
independent_variables = dataset[['age']]
dependent_variable = dataset['bought_insurance']
print(independent_variables)
print(dependent_variable)

independent_variables_training_dataset_subset, independent_variables_test_dataset_subset, dependent_variable_training_dataset_subset, dependent_variable_test_dataset_subset = train_test_split(independent_variables, dependent_variable, test_size=0.1, random_state=10)

print(f'The independent variables training dataset subset: \n{independent_variables_training_dataset_subset}')

print(f'The independent variables test dataset subset: \n{independent_variables_test_dataset_subset}')

print(f'The dependent variables training dataset subset: \n{dependent_variable_training_dataset_subset}')

print(f'The dependent variables test dataset subset: \n{dependent_variable_test_dataset_subset}')


# //////////////////////////////////////////////////////////////////////////////


# Using the Scikit-learn Machine Learning (ML) Python library, we will now use its BLR ML algorithm class (refer to the 
# '3. About_scikit-learn_Machine_Learning_Python_library.txt' file in the 'Tutorial 1 - What is Machine Learning' folder),
#       'LogisticRegression()' from the 'sklearn.linear_model' Scikit-learn ML Python library's submodule 

# to create a 'BLR ML model' class object/instance, and allowing us to access the 
# BLR ML algorithm class's Instance Methods such as:
#    - '.fit()'           - this Instance Method trains the BLR ML algorithm to become a BLR ML model
#    - '.predict()'       - this Instance Method makes a prediction using the BLR ML model
#    - '.predict_proba()' - this Instance Method returns the predicted probability of the prediction 
#                           being in each category of the categorical dependent variable
#    - '.score()'         - this Instance Method evaluates the accuracy of the BLR ML model

# and the BLR ML algorithm class's attributes such as (note that the trailing underscore ('_') is a convention 
# used in the Scikit-learn Machine Learning (ML) Python library to represent attributes):
#    - '.coef_'      - this attribute represents the values of the weights/gradients/coefficients of the independent 
#                      variables/features 'x1', 'x2', 'x3' ... 'x? , 'm1', 'm2', 'm3' ... 'm?' respectively, in 
#                      the '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b' in the mathematical equation representing the 
#                      BLR ML algorithm, 'sigmoid(z) = 1 / (1 + 𝑒^−z)'  
#    - '.intercept_' - this attribute represents the value of the bias/intercept 'b', in the 
#                      '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b' in the mathematical equation representing 
#                      the BLR ML algorithm, 'sigmoid(z) = 1 / (1 + 𝑒^−z)'


# Creating a 'BLR ML model' class object/instance
binary_logistic_regression_ML_model = LogisticRegression()



# Training the 'BLR ML model' class object/instance

# For the '.fit(X, y, sample_weight=None)' Instance Method of the 'BLR ML model' class, 
# it takes 3 parameters, with the first 2 parameters are the more important ones:
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the BLR ML algorithm
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      BLR ML algorithm
print(binary_logistic_regression_ML_model.fit(independent_variables_training_dataset_subset, dependent_variable_training_dataset_subset))



# The '.coef_' attribute of the 'BLR ML model' class shows the values of the weights/gradients/coefficients of the 
# independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1', 'm2', 'm3' ... 'm?' respectively, 
# in the '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b' in the mathematical equation representing the 
# BLR ML algorithm, 'sigmoid(z) = 1 / (1 + 𝑒^−z)'

# (Note: The multiple coefficient values of the independent variables/features, 'm1', 'm2', 'm3' ... 'm?', will be displayed
#        in the output in order, in a Python list, for example, '[   137.25 -26025.    -6825.  ]', where '137.25' will be 
#        'm1', '-26025' will be 'm2', and '-6825' will be 'm3')
print(binary_logistic_regression_ML_model.coef_)                        # output: [[0.12740563]]



# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the bias/intercept, 'b', in the 
# '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b' in the mathematical equation representing the BLR ML algorithm, 
# 'sigmoid(z) = 1 / (1 + 𝑒^−z)'
print(binary_logistic_regression_ML_model.intercept_)                   # output: -4.97335111



# Making predictions with the 'BLR ML model' class object/instance

# For the '.predict(X)' Instance Method of the 'BLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'BLR ML model' class object/instance to make a prediction of. The output of this Instance Method is the predicted
#      binary categorical dependent variable, 'y', of the independent variables/features by the 'BLR ML model' class 
#      object/instance

# Hence, in this case, 
#   -> the output of this '.predict()' Instance Method of the 'BLR ML model' class is the (binary categorical) dependent
#      variable, 'y' '[1 1 0]', using the mathematical equation representing the BLR ML algorithm, 
#      'sigmoid(z) = 1 / (1 + 𝑒^−z)', where '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b', and the probability threshold, 
#      which is usually 0.5 for BLR ML algorithm, and:
#      => if the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', probability is larger than the probability 
#         threshold of 0.5, then the (predicted) 'y' (binary categorical) dependent variable value for a set of values 
#         of the independent variables/features for the BLR ML algorithm will be rounded up to 1, which can be interpreted 
#         as the (predicted) 'y' (binary categorical) dependent variable value to be the category represented by the value 
#         of 1
#      => if the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', probability is smaller than the probability 
#         threshold of 0.5, then the (predicted) 'y' (binary categorical) dependent variable value for a set of values of 
#         the independent variables/features for the BLR ML algorithm will be rounded down to 0, which can be interpreted 
#         as the (predicted) 'y' (binary categorical) dependent variable value to be the category represented by the value 
#         of 0

#   -> the input 'X' parameter of this '.predict()' Instance Method of the 'BLR ML model' class is the independent 
#      variables/features, 'x1', 'x2', 'x3' in the mathematical equation representing the BLR ML algorithm, 
#      'sigmoid(z) = 1 / (1 + 𝑒^−z)', where '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b'

# Note: the multiple independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the '.predict(X)' 
# Instance Method of the 'BLR ML model' class will need to be in order, in a 2D array
print(binary_logistic_regression_ML_model.predict(independent_variables_test_dataset_subset))            # output: [1 1 0]

# These are the 3 sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
# '.predict(X)' Instance Method of the 'BLR ML model' class that we are using to make 3 respective predictions using the 
# BLR ML model
print(independent_variables_test_dataset_subset)
# Output:
#     age
# 7    60
# 5    56
# 18   19


# Interpreting the predicted 'y' (binary categorical) dependent variable values using the BLR ML model for this dataset:
# For each of the 3 sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
# '.predict(X)' Instance Method of the 'BLR ML model' class in the 'independent_variables_test_dataset_subset' variable, of 
# 'age = 60', 'age = 56', and 'age = 19', and since for this 'bought_insurance_or_not_bought_insurance_with_single_
# independent_variable_and_binary_categorical_dependent_variableble.csv' dataset a 'y' (binary categorical) dependent 
# variable value of 0 represents the category 'not bought insurance', while a 'y' (binary categorical) dependent variable 
# value of 1 represents the category 'bought insurance',

# hence, the 3 respective predicted 'bought insurance or not bought insurance' ('y') (binary categorical) dependent 
# variable value using the BLR ML model are:
# - '1', representing the 'bought insurance' category for 'age = 60'
# - '1', representing the 'bought insurance' category for 'age = 56'
# - '0', representing the 'not bought insurance' category for 'age = 19'



# The predicted probability of the predictions with the 'BLR ML model' class object/instance being in each category of 
# the categorical dependent variable
# For the '.predict_proba(X)' Instance Method of the 'BLR ML model' class, it takes 1 parameter:
#    - 'X' is a parameter of a 2D array that represents the test independent variables/features you want the
#      'BLR ML model' class object/instance to return the predicted probability of the prediction with the 
#      'BLR ML model' class object/instance being in each category of the categorical dependent variable. The output 
#      of this Instance Method is the predicted probability of the prediction with the 'BLR ML model' class 
#      object/instance being in each category of the categorical dependent variable, returned as a 2D array

# Note: The categorical dependent variable labels each category with a value in the dataset, for example, in this 
#       'bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_dependent_
#       variable.csv' dataset, the 'bought insurance' category is represented by the value of 1, while the 'not bought 
#       insurance' category is represented by the value of 0.

#       Hence, the returned 2D array of the predicted probability of the prediction with the 'BLR ML model' class 
#       object/instance being in each category of the categorical dependent variable will order the category of the 
#       categorical dependent variable represented by the smallest value as the first index of the 1D arrays in the
#       returned 2D array, and the category of the categorical dependent variable represented by the second smallest 
#       value as the second index of the 1D arrays in the returned 2D array, the category of the categorical dependent 
#       variable represented by the represented by the third smallest value as the third index of the 1D arrays in 
#       the returned 2D array, and so on...

#       For example, for the 3 sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input 
#       parameters in the '.predict(X)' Instance Method of the 'BLR ML model' class in the 'independent_variables_test_
#       dataset_subset' variable by the 'BLR ML model' class object/instance of the 'bought_insurance_or_not_bought_
#       insurance_with_single_independent_variable_and_binary_categorical_dependent_variable_variable.csv' dataset, 
#       from the output of this '.predict_proba()' Instance Method, 
#           [[0.06470723 0.93529277]
#            [0.10327405 0.89672595]
#            [0.92775095 0.07224905]]

#       since the 'not bought insurance' category of the categorical dependent variable represented by the smallest 
#       value (of 0) is the first index of the 1D arrays in the returned 2D array, and the 'bought insurance' category 
#       of the categorical dependent variable represented by the second smallest value (of 1) as the second index of 
#       the 1D arrays in the returned 2D array, hence, the 3 respective predicted probability of the predictions 
#       of the 3 sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
#       '.predict(X)' Instance Method of the 'BLR ML model' class in the 'independent_variables_test_dataset_subset' 
#       variable, of 'age = 60', 'age = 56', and 'age = 19' are:
#       - for 'age = 60', there is a predicted probability of 0.06470723, for the first index of the 1D arrays in the 
#         returned 2D array, which represents the 'not bought insurance' category, which is the categorical dependent 
#         variable with the smallest value of 0 and a predicted probability of 0.93529277, for the second index of the 
#         1D arrays in the returned 2D array, which represents 'bought insurance' category, which is the categorical 
#         dependent variable with the second smallest value of 1
#       - for 'age = 56', there is a predicted probability of 0.10327405, for the first index of the 1D arrays in the 
#         returned 2D array, which represents the 'not bought insurance' category, which is the categorical dependent 
#         variable with the smallest value of 0 and a predicted probability of 0.89672595, for the second index of the 
#         1D arrays in the returned 2D array, which represents 'bought insurance' category, which is the categorical 
#         dependent variable with the second smallest value of 1
#       - for 'age = 19', there is a predicted probability of 0.92775095, for the first index of the 1D arrays in the 
#         returned 2D array, which represents the 'not bought insurance' category, which is the categorical dependent 
#         variable with the smallest value of 0 and a predicted probability of 0.07224905, for the second index of the 
#         1D arrays in the returned 2D array, which represents 'bought insurance' category, which is the categorical 
#         dependent variable with the second smallest value of 1
# 
#       And since, recalling that the since the BLR ML algorithm uses the probability threshold, that is usually 0.5,
#       to determine the (predicted) 'y' (binary categorical) dependent variable value for a set of values of the 
#       independent variables/features (see the section 'How does the Binary Logistic Regression (BLR) Machine Learning 
#       (ML) algorithm work' in the '1. What_is_Logistic_Regression_and_Binary_Logistic_Regression.txt' file), 
#       - for 'age = 60', the predicted probability is 0.06470723 for the 'not bought insurance' category represented
#         by a value of 0, and a predicted probability of 0.93529277 for the 'bought insurance' category represented
#         by a value of 1, the (predicted) 'y' (binary categorical) dependent variable value will be 1 due to the 
#         'bought insurance' category represented by a value of 1 having a higher predicted probability that is larger
#         than the probability threshold, that is usually 0.5
#       - for 'age = 56', the predicted probability is 0.10327405 for the 'not bought insurance' category represented
#         by a value of 0, and a predicted probability of 0.89672595 for the 'bought insurance' category represented
#         by a value of 1, the (predicted) 'y' (binary categorical) dependent variable value will be 1 due to the 
#         'bought insurance' category represented by a value of 1 having a higher predicted probability that is larger
#         than the probability threshold, that is usually 0.5
#       - for 'age = 19', the predicted probability is 0.92775095 for the 'not bought insurance' category represented
#         by a value of 0, and a predicted probability of 0.07224905 for the 'bought insurance' category represented
#         by a value of 1, the (predicted) 'y' (binary categorical) dependent variable value will be 0 due to the 
#         'not bought insurance' category represented by a value of 0 having a higher predicted probability that is larger
#         than the probability threshold, that is usually 0.5

#       (Note: You can see that this aligns with the (predicted) 'y' (binary categorical) dependent variable values for 
#        the 3 sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
#        '.predict(X)' Instance Method of the 'BLR ML model' class in the 'independent_variables_test_dataset_subset'
#        of '[1 1 0]')
print(binary_logistic_regression_ML_model.predict_proba(independent_variables_test_dataset_subset))
# Output:
# [[0.06470723 0.93529277]
#  [0.10327405 0.89672595]
#  [0.92775095 0.07224905]]



# //////////////////////////////////////////////////////////////////////////


# Continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to evaluate the 
# performance of the trained BLR ML model (refer to the 'Tutorial 12 - Train Test Split Machine Learning Model 
# Evaluation Technique (Machine Learning Technique)' folder for more on the Train Test Split Machine Learning 
# (ML) Model Evaluation Technique and explanation on the code).

# Evaluating the trained BLR ML model using the '.score()' Instance Method of the 'Binary Logistic Regression 
# (BLR) ML model' class, using the test dataset subset obtained from the Train Test Split ML Model Evaluation Technique

# Apart from the the BLR ML algorithm class giving us access to the '.fit()' and '.predict()' Instance Methods and the
# '.coef_' and '.intercept_' attributes, the BLR ML algorithm class also gives us access to the '.score()' Instance
# Method, which allows us to calculate how accurate the BLR ML model is:

# How the BLR ML algorithm class's '.score(X, y)' Instance method calculate how accurate our BLR ML model is,
# is that it takes in 2 parameters, 
#    - 'X' is a parameter of a 2D array that represents the independent variables/features of the dataset used to 
#      train the BLR ML algorithm (In this tutorial, this will be the 'independent_variables_test_dataset_subset'
#      obtained from the Train Test Split ML Model Evaluation Technique)
#    - 'y' is a parameter of a 1D array that represents the dependent variables of the dataset used to train the 
#      BLR ML algorithm (In this tutorial, this will be the 'dependent_variables_test_dataset_subset'
#      obtained from the Train Test Split ML Model Evaluation Technique)

# It then uses the BLR ML model, which is trained with the 'independent_variables_training_dataset_subset', to 
# predict each of the predicted 'bought insurance or not bought insurance' categorical dependent variable for each 
# of the independent variables/features in the 'independent_variables_test_dataset_subset', and compare them with the 
# provided actual predicted 'bought insurance or not bought insurance' categorical dependent variable in the 
# 'dependent_variable_test_dataset_subset'. It then uses some mathematical function (not super important honestly)
# to calculate the percentage (%) closeness of the predicted predicted 'bought insurance or not bought insurance' 
# categorical dependent variable by the BLR ML model, which is trained with the 'independent_variables_training_
# dataset_subset', with the provided actual predicted 'bought insurance or not bought insurance' categorical dependent 
# variable in the 'dependent_variable_test_dataset_subset', as an indicator score for how accurate the BLR ML model is 
print(binary_logistic_regression_ML_model.score(independent_variables_test_dataset_subset, dependent_variable_test_dataset_subset))
# Output: 1.0

print(f'    bought_insurance\n{dependent_variable_test_dataset_subset}')
# Output:
#     bought_insurance
# 7     1
# 5     1
# 18    0


# Interpreting the evaluation accuracy score of the BLR ML model for this dataset:
# The evaluation accuracy score of the BLR ML model is 1.0 for this dataset, meaning that our BLR ML model is interpreted
# as perfect. But this only happens since our dataset size is small, and is usually not the case.

# Our BLR ML model is interpreted as perfect because the predicted 'y' (binary categorical) dependent variable value for
# the independent variables/features in the 'independent_variables_training_dataset_subset' of '[1 1 0]' (see the output 
# of the '.predict()' Instance Method above) and the provided actual predicted 'bought insurance or not bought insurance' 
# (categorical dependent variable) in the 'dependent_variable_test_dataset_subset' of '[1 1 0]' are both exactly the same.

# But if we use larger datasets, the BLR ML model is bound to make errors in its predicted 'y' (binary categorical) 
# dependent variable value for the independent variables/features that will not be the same as the provided actual 
# predicted categorical dependent variable, and will cause the evaluation accuracy score to be less than 1.0.


# //////////////////////////////////////////////////////////////////////////


# Some key observations from all this:

# So how is this predicted value/output made by the 'Binary Logistic Regression (BLR) ML model' class 
# object/instance?
# Essentially, looking at in the mathematical equation representing the BLR ML algorithm, 
# 'sigmoid(z) = 1 / (1 + 𝑒^−z)', where '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b', since:
# - the '.coef_' attribute of the 'BLR ML model' class shows the value of the weight/gradient/coefficient of the 
#   independent variables/features 'x', is 'm' ('0.12740563') 
# - the input 'X' parameter of this '.predict()' Instance Method of the 'BLR ML model' class is the independent 
#   variables/features, ('age') 'x' ('60')
# - the '.intercept_' attribute of the 'BLR ML model' class shows the value of the bias/intercept 'b' ('-4.97335111')
# - '𝑒' is the Euler's number ~ 2.71828

# z =  'm1'    * 'x1' + ... ('m2'  *  'x2' +  'm3'  * 'x3') + ... + 'b'
z = 0.12740563 *  60                                              - 4.97335111 

#     1 / (1 +     𝑒    ^ −'z')         = 'sigmoid(z)'
print(1 / (1 + 2.71828 ** (-z)))      # = 0.9352926629922721

# And since the output of the sigmoid/logit function is the mapped real number of z to a value between 0 and 1 
# is so that the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', can be interpreted as a probability!

# Using a probability threshold, which is usually 0.5 for BLR ML algorithm and since 'y' is the (binary categorical) 
# dependent variable, which is the output of the '.predict()' Instance Method of the 'BLR ML model' class, 
# -> if the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', probability is larger than the probability threshold 
#    of 0.5, then the (predicted) 'y' (binary categorical) dependent variable value for a set of values of the independent 
#    variables/features for the BLR ML algorithm will be rounded up to 1, which can be interpreted as the (predicted) 'y' 
#    (binary categorical) dependent variable value to be the category represented by the value of 1
# -> if the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', probability is smaller than the probability threshold 
#    of 0.5, then the (predicted) 'y' (binary categorical) dependent variable value for a set of values of the independent 
#    variables/features for the BLR ML algorithm will be rounded down to 0, which can be interpreted as the (predicted) 'y' 
#    (binary categorical) dependent variable value to be the category represented by the value of 0

# Hence, since the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', probability is larger than the probability 
# threshold of 0.5, the (predicted) 'y' (binary categorical) dependent variable value for a set of values of the independent 
# variables/features for the BLR ML algorithm will be rounded up to 1
print(round(1 / (1 + 2.71828 ** (-z))))     # Output: 1




# Also, as you can see, when we try to plot a sigmoid/logit line graph through the output of the sigmoid/logit function,
# which is the mapped real number of z to a value between 0 and 1, the real-number-of-z-now-a-value-between-0-and-1, 
# 'sigmoid(z)', and the corresponding 'age' independent variable/feature on the same scatter graph as the dataset that 
# we used to train the 'BLR ML model', you can see that the sigmoid/logit line graph through the output of the 
# sigmoid/logit function, which is the mapped real number of z to a value between 0 and 1, the 
# real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)' and the corresponding 'age' independent
# variable/feature is actually a best fit sigmoid/logit line through the data points of the dataset that we used to 
# train the 'BLR ML model'.

# Since the sigmoid/logit line graph through the output of the sigmoid/logit function, which is the mapped real number 
# of z to a value between 0 and 1, the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)' and the corresponding 
# 'age' independent variable/feature represents the 'BLR ML model', this shows that the BLR ML algorithm is essentially 
# the best fit sigmoid/logit line (of mathematical equation representing the BLR ML algorithm, 
# 'sigmoid(z) = 1 / (1 + 𝑒^−z)', where '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b') through the dataset's data points
plt.title("Matplotlib graphical visualisation of the Binary Logistic\nRegression (BLR) Machine Learning (ML) model")
dataset_sorted = dataset.sort_values('age')
predictions = binary_logistic_regression_ML_model.predict_proba(dataset_sorted[['age']])[:, 1]
plt.plot(dataset_sorted['age'], predictions, color='blue')

plt.savefig('matplotlib_visualisation_of_the_binary_logistic_regression_ML_model.png', dpi=100)
plt.show()


# Note: 
# - Why doesn't 'plt.plot(dataset['age'], binary_logistic_regression_ML_model.predict(dataset[['age']]), color='blue')'
#   work (it plots some weird looking line instead of the 'S' shaped sigmoid/logit line graph)? 
#   This is because you are required to first sort the dataset based on its independent variable/feature, and plot
#   the 'S' shaped sigmoid/logit line graph using the '.predict_proba()' Instance Method since it returns the 2D array
#   output, where the second index of the returned 2D array of the sigmoid/logit function, is the mapped real number 
#   of z to a value between 0 and 1, the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', while the 
#   '.predict()' Instance Method only returns either a value of 0 or 1.

# - Why do I need to first sort the dataset based on its independent variable/feature? 
#   (From: dataset_sorted = dataset.sort_values('age'))
#   (From ChatGPT Im lazy type...):
#   The dataset is often sorted when plotting the BLR ML model's curve to ensure that the graph displays a smooth, 
#   continuous sigmoid curve rather than a scattered, disordered plot.

#   Here's why:
#   Sigmoid Curve Appearance: BLR ML model's models a sigmoid-shaped probability curve. Sorting the dataset by 
#   the feature (in your case, age) ensures that the corresponding predicted probabilities also follow an increasing 
#   (or decreasing) order. This helps plot the typical "S-shaped" curve of BLR ML model.

#   Smooth Plotting: If the data points are not sorted, the plot may appear jagged or disjointed, because the x-axis 
#   (age) would not be in a logical sequence. Sorting helps align the data along the x-axis (from lowest to highest), 
#   making the curve appear smooth and continuous.

#   Interpretability: Sorting helps in interpreting the graph visually. When you look at the plot, you can easily 
#   follow how the predicted probabilities change as age increases, making it more intuitive to understand the 
#   relationship between the feature (age) and the outcome.

#   So, sorting ensures that the plotted BLR ML model's curve correctly follows the shape that corresponds to the 
#   ordered input values, giving a clear and smooth representation of the model's behavior.

# - Why is the second index of the returned 2D array of the sigmoid/logit function, is the mapped real number 
#   of z to a value between 0 and 1, the real-number-of-z-now-a-value-between-0-and-1, 'sigmoid(z)', and not
#   the first index? (From 'predictions = binary_logistic_regression_ML_model.predict_proba(dataset_sorted[['age']])[:, 1]')
#   (From ChatGPT Im lazy type...):
#   In BLR ML models, the BLR ML model typically predicts probabilities for each class in a binary classification 
#   problem. The '.predict_proba()' method returns a 2D array with two columns:
#       Index 0: The predicted probability for the negative class (e.g., class 0).
#       Index 1: The predicted probability for the positive class (e.g., class 1).

#   Since you are interested in plotting the probability of the positive class (e.g., the event of interest, 
#   often labeled as class 1), you use [:, 1] to select the probabilities for that class.

#   If you were to use [:, 0], you would be plotting the probability of the negative class (class 0), which is 
#   complementary to the probability of the positive class (i.e., 1 - probability of class 1).


