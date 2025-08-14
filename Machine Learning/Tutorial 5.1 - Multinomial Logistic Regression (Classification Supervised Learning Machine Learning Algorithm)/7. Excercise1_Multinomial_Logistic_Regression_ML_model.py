# Question 1:
# Use Scikit-learn's iris flower dataset to train a Multinomial Logistic Regression (MNLR) Machine
# Learning (ML) model. You need to figure out accuracy of your model and use that to predict different 
# samples in your test dataset. 
# In Scikit-learn's iris flower dataset there are 150 flower samples containing following features,
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width

# Using the above 4 features you will clasify a flower in one of the three categories,
# 1. Setosa
# 2. Versicolour
# 3. Virginica

# (Hint: Basically the task here is to predict a flower category based on information of a flower's 
# features)



# About the Scikit-learn Digits Image Dataset:
# The Scikit-learn library in Python comes with a number of built-in datasets, one of which is the 
# Iris flower image dataset. This dataset contains 150x4 numpy.ndarray, with the rows being the sets of 
# information and the columns being the flower features: Sepal Length, Sepal Width, Petal Length and Petal 
# Width. It is often used for classification tasks.

# It works by having each flower feature represent an independent variable/feature. Hence there will be 4 
# independent variable/features in this iris flower dataset due to the 4 different flower features, Sepal 
# Length, Sepal Width, Petal Length and Petal Width. 
# (Note: each of the 4 pixel/independent variable/feature, 'x1', 'x2', 'x3' ... 'x?' will have their own 
#  respective weights/gradients/coefficients, 'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)')
 
# The nominal categorical dependent variable will be the flower type categories. Hence there will be
# 3 nominal categorical dependent variable categories, with each category representing a flower type
# that has the corresponding flower features, and there is 3 different flower types Setosa, Versicolour 
# and Virginica.
# (Note: each of the 3 nominal categorical dependent variable categories will have their own 
#  respective bias/intercept, 'bi')

# Documentation: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Importing Scikit-learn's iris flower dataset from the Scikit-learn Machine Learning (ML) library
from sklearn.datasets import load_iris

import seaborn as sn
from sklearn.metrics import confusion_matrix


# Reading the iris flower dataset
iris_flower_dataset = load_iris()
print(iris_flower_dataset)


# It works by having each flower feature represent an independent variable/feature. Hence there will be 4 
# independent variable/features in this iris flower dataset due to the 4 different flower features, Sepal 
# Length, Sepal Width, Petal Length and Petal Width. 
# (Note: each of the 4 pixel/independent variable/feature, 'x1', 'x2', 'x3' ... 'x?' will have their own 
#  respective weights/gradients/coefficients, 'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)')
 
# The nominal categorical dependent variable will be the flower type categories. Hence there will be
# 3 nominal categorical dependent variable categories, with each category representing a flower type
# that has the corresponding flower features, and there is 3 different flower types Setosa, Versicolour 
# and Virginica.
# (Note: each of the 3 nominal categorical dependent variable categories will have their own 
#  respective bias/intercept, 'bi')
# Using the 'dir()' function to see the available attributes of the iris flower dataset
# There are 7 available attributes of the iris flower dataset, but the most important 4 attributes
# are:
# - 'data'          - it contains the values of the independent variables/features of the iris flower 
#                     dataset, where each independent variable/feature represents a flower feature, and 
#                     there will be 4 independent variable/features in this iris flower dataset due to 
#                     the 4 different flower features, Sepal Length, Sepal Width, Petal Length and Petal 
#                     Width. 
# - 'target'        - it contains the nominal categorical dependent variable of the iris flower dataset,
#                     where each nominal category representing the flower type, and there will be 3 
#                     nominal categories of the nominal categorical dependent variable in this iris 
#                     flower dataset due to each category representing a flower type that has the 
#                     corresponding flower features, and there is 3 different flower types Setosa, 
#                     Versicolour and Virginica.
# - 'feature_names' - it contains the list of the names of the features: ['sepal length (cm)', 
#                     'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']. These feature names 
#                     correspond to the columns index in the data array in order and represent the physical 
#                     characteristics of the iris flowers sets of information.
# - 'target_names'  - it contains the list of the unique flower types: ['setosa', 'versicolor', 'virginica'], 
#                     representing the 3 type of iris flowers. The indices in this list corresponding to 
#                     the list in the 'target' attribute's list, where:
#                     => Index 0 represents 'setosa' flower type
#                     => Index 1 represents 'versicolor' flower type
#                     => Index 2 represents 'virginica' flower type
print(dir(iris_flower_dataset))               # output: ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']


# Exploring the first 5 sets of information in the iris flower dataset:

# Exploring the first 5 sets of the 'data' attribute of the iris flower dataset (independent 
# variables/features of the iris flower dataset)
print(iris_flower_dataset.data[0:5])

# Exploring the first 5 sets of the 'target' attribute of the iris flower dataset (nominal 
# categorical dependent variable of the iris flower dataset)
print(iris_flower_dataset.target[0:5])

# Exploring the 'target_names' attribute of the iris flower dataset
print(iris_flower_dataset.target_names)               # output: ['setosa' 'versicolor' 'virginica']

# Exploring the 'feature_names' attribute of the iris flower dataset
print(iris_flower_dataset.feature_names)              # output: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# ///////////////////////////////////////////////////////////////////////


# Using the Train Test Split Machine Learning (ML) Model Evaluation Technique to divide the dataset into two 
# seperate subsets: the training dataset subset and the test dataset subset (refer to the 'Tutorial 12 - Train 
# Test Split Machine Learning Model Evaluation Technique (Machine Learning Technique)' folder for more on the 
# Train Test Split Machine Learning (ML) Model Evaluation Technique and explanation on the code).

# Doing the Train Test Split Machine Learning (ML) Model Evaluation Technique using Scikit-learn's 'train_test_split' 
# Instance Method in the 'sklearn.model_selection' class
independent_variables = iris_flower_dataset.data
dependent_variable = iris_flower_dataset.target
print(independent_variables)
print(dependent_variable)

independent_variables_training_dataset_subset, independent_variables_test_dataset_subset, dependent_variable_training_dataset_subset, dependent_variable_test_dataset_subset = train_test_split(independent_variables, dependent_variable, test_size=0.2, random_state=10)

print(f'The independent variables training dataset subset: \n{independent_variables_training_dataset_subset}')

print(f'The independent variables test dataset subset: \n{independent_variables_test_dataset_subset}')

print(f'The dependent variables training dataset subset: \n{dependent_variable_training_dataset_subset}')

print(f'The dependent variables test dataset subset: \n{dependent_variable_test_dataset_subset}')


# We will later continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to 
# evaluate the performance of the MNLR ML model once it is trained.


# ///////////////////////////////////////////////////////////////////////


# Creating a 'MNLR ML model' class object/instance
multinomial_logistic_regression_ML_model = LogisticRegression()


# Training the 'MNLR ML model' class object/instance
print(multinomial_logistic_regression_ML_model.fit(independent_variables_training_dataset_subset, dependent_variable_training_dataset_subset))


# The '.coef_' attribute of the 'MNLR ML model' class shows the values of the weights/gradients/coefficients of the 
# independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)' 
# respectively, in the '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi', 
# for 𝑖 = 1, 2, … , 𝐾  in the mathematical equation representing the MNLR ML algorithm, 
# 'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)'
print(multinomial_logistic_regression_ML_model.coef_)


# The '.intercept_' attribute of the 'MNLR ML model' class shows the value of the bias/intercept, 'bi', in the 
# '𝑧i = m(zi, 1) * x1 + m(zi, 2) * x2 + m(zi, 3) * x3 + ... + m(zi, ?) * x? + bi' for 𝑖 = 1, 2, … , 𝐾  in the 
# mathematical equation representing the MNLR ML algorithm, 'softmax(𝑧𝑖) = 𝑒^𝑧𝑖 / (𝑒^𝑧1 + 𝑒^𝑧2 + ... + 𝑒^𝑧k)'
print(multinomial_logistic_regression_ML_model.intercept_)


# Making predictions with the 'MNLR ML model' class object/instance
print(multinomial_logistic_regression_ML_model.predict(independent_variables_test_dataset_subset))

# These are the sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
# '.predict(X)' Instance Method of the 'MNLR ML model' class that we are using to make the respective predictions using the 
# MNLR ML model
print(independent_variables_test_dataset_subset)


# The predicted probability of the predictions with the 'MNLR ML model' class object/instance being in each category of 
# the categorical dependent variable
print(multinomial_logistic_regression_ML_model.predict_proba(independent_variables_test_dataset_subset))


# //////////////////////////////////////////////////////////////////////////


# Continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to evaluate the 
# performance of the trained MNLR ML model (refer to the 'Tutorial 12 - Train Test Split Machine Learning Model 
# Evaluation Technique (Machine Learning Technique)' folder for more on the Train Test Split Machine Learning 
# (ML) Model Evaluation Technique and explanation on the code).

# Evaluating the trained MNLR ML model using the '.score()' Instance Method of the 'Multinomial Logistic Regression 
# (MNLR) ML model' class, using the test dataset subset obtained from the Train Test Split ML Model Evaluation Technique
print(multinomial_logistic_regression_ML_model.score(independent_variables_test_dataset_subset, dependent_variable_test_dataset_subset))
# Output: 1.0

print(f'     Iris Flower Category\n{dependent_variable_test_dataset_subset}')


# //////////////////////////////////////////////////////////////////////////////////


# Testing random sets of information in the iris flower dataset with the MNLR ML model:
# Testing if the predicted 'y' nominal categorical dependent variable by the MNLR ML model is the same as the
# actual 'y' nominal categorical dependent variable

# The actual 'y' nominal categorical dependent variable of the set of information in the iris flower 
# dataset at index 67 with the 'target' attribute
print(iris_flower_dataset.target[67])            # output: 1

print(iris_flower_dataset.target_names[iris_flower_dataset.target[67]])            # output: versicolor

# # The predicted 'y' nominal categorical dependent variable by the MNLR ML model of the set of information 
# in the iris flower dataset at index 67
print(multinomial_logistic_regression_ML_model.predict([iris_flower_dataset.data[67]]))             # output: 1

print(iris_flower_dataset.target_names[multinomial_logistic_regression_ML_model.predict([iris_flower_dataset.data[67]])])            # output: versicolor



# The actual 'y' nominal categorical dependent variable of the set of information in the iris flower 
# dataset at index 1 with the 'target' attribute
print(iris_flower_dataset.target[1])            # output: 0

print(iris_flower_dataset.target_names[iris_flower_dataset.target[1]])            # output: setosa

# # The predicted 'y' nominal categorical dependent variable by the MNLR ML model of the set of information 
# in the iris flower dataset at index 1
print(multinomial_logistic_regression_ML_model.predict([iris_flower_dataset.data[1]]))             # output: 0

print(iris_flower_dataset.target_names[multinomial_logistic_regression_ML_model.predict([iris_flower_dataset.data[1]])])            # output: setosa



# The actual 'y' nominal categorical dependent variable of the set of information in the iris flower 
# dataset at index 130 with the 'target' attribute
print(iris_flower_dataset.target[130])            # output: 2

print(iris_flower_dataset.target_names[iris_flower_dataset.target[130]])            # output: virginica

# # The predicted 'y' nominal categorical dependent variable by the MNLR ML model of the set of information 
# in the iris flower dataset at index 130
print(multinomial_logistic_regression_ML_model.predict([iris_flower_dataset.data[130]]))             # output: 2

print(iris_flower_dataset.target_names[multinomial_logistic_regression_ML_model.predict([iris_flower_dataset.data[130]])])            # output: virginica


# //////////////////////////////////////////////////////////////////////////////////
 

# Visualising the Confusion Matrix of the MNLR ML model with the iris flower dataset:
predictions = multinomial_logistic_regression_ML_model.predict(independent_variables_test_dataset_subset)
confusion_matrix_of_multinomial_logistic_regression_ML_model = confusion_matrix(dependent_variable_test_dataset_subset, predictions)

print(confusion_matrix_of_multinomial_logistic_regression_ML_model)
# output:
# [[10  0  0]
#  [ 0 13  0]
#  [ 0  0  7]]


# Representing the Confusion Matrix in a more presentable manner using the Seaborn (Python) library's 
# 'heatmap()' function and Matplotlib (Python) library

# The 'plt.figure()' Matplotlib (Python) library function is optional, it just defines the dimensions/size
# of the plot (which the heatmap plot is defined with dimensions 10 inches by 7 inches in this context)
plt.figure(figsize = (10,7))
plt.title('Confusion Matrix for Multinomial Logistic Regression Machine Learning (ML)\nmodel on Scikit-learn Iris Flower Dataset')
plt.xlabel('predicted "y" categorical dependent variable category by the MNLR ML model')
plt.ylabel('actual "y" categorical dependent variable category')


# The 'sn.heatmap()' Seaborn (Python) library function creates a heatmap, and takes in 12 parameters, 
# with the first 2 parameters being the more important ones:
# - 'data'  - a 2D dataset to plot the heatmap with 
# - 'annot' - is set to 'True' by default. If 'True', the heatmap will have the data value, along with 
#             the colors representing the values in each cell, else if 'False', the heatmap will not 
#             have the data value in each cell, but will still have the colors representing the values 
#             in each cell
sn.heatmap(confusion_matrix_of_multinomial_logistic_regression_ML_model, annot=True)


plt.savefig('confusion_matrix_visualisation_of_the_Multinomial_Logistic_Regression_ML_model_on_the_scikit-learn_iris_flower_dataset.png', dpi=100)
plt.show()



# My answers are all correct except for part 1, where I did not know how to do and had to refer to the
# solution for answer.