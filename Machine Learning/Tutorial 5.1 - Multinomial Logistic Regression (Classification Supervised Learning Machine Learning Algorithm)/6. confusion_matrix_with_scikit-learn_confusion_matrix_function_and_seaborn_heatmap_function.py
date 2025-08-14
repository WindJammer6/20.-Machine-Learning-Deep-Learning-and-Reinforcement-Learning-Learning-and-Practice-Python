# About the Scikit-learn Digits Image Dataset:
# The Scikit-learn library in Python comes with a number of built-in datasets, one of which is the 
# Digits image dataset. This dataset contains 1797 8x8 pixel images of handwritten digits (0-9). It is 
# often used for classification tasks, particularly for algorithms that classify digits based on 
# the pixel data.

# It works by having each pixel represent an independent variable/feature. Hence there will be 64 independent 
# variable/features in this digits image dataset due to the size of each image being 8x8 pixels. The value
# of each pixel will determine the color shade of the pixel (in grayscale, a smaller value represents a lighter
# shade, while a larger value represents a darker shade). 
# (Note: each of the 64 pixel/independent variable/feature, 'x1', 'x2', 'x3' ... 'x?' will have their own 
#  respective weights/gradients/coefficients, 'm(zi, 1)', 'm(zi, 2)', 'm(zi, 3)' ... 'm(zi, ?)')
 
# The nominal categorical dependent variable will be the digit image categories. Hence there will be
# 10 nominal categorical dependent variable categories, with each category representing the digit of
# 0 to 9 the image of pixels is supposed to be showing.
# (Note: each of the 10 nominal categorical dependent variable categories will have their own 
#  respective bias/intercept, 'bi')

# Documentation: https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Importing Scikit-learn's digits image dataset from the Scikit-learn Machine Learning (ML) library
from sklearn.datasets import load_digits

import seaborn as sn
from sklearn.metrics import confusion_matrix


# Reading the digits image dataset
digits_image_dataset = load_digits()
print(digits_image_dataset)

# Using the 'dir()' function to see the available attributes of the digits image dataset
print(dir(digits_image_dataset))               # output: ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']


# Exploring the first 5 sets of information in the digits image dataset:

# Exploring the first 5 sets of the 'data' attribute of the digits image dataset (independent 
# variables/features of the digits image dataset)
print(digits_image_dataset.data[0:5])

# Exploring the first 5 sets of the 'target' attribute of the digits image dataset (nominal 
# categorical dependent variable of the digits image dataset)
print(digits_image_dataset.target[0:5])

# Exploring the first 5 sets of the 'image' attribute of the digits image dataset (actual 8x8 
# pixel images of the digits in the form of a 2D array of shape (8,8)), and then using Matplotlib 
# (Python) library's 'matshow()' function to visualise these images (convert the 2D array of shape 
# (8,8) to an image)
plt.gray()
for i in range(5):
    plt.matshow(digits_image_dataset.images[i])
    plt.title(f'Digits image with the index {i} in the\nScikit-learn Digits image dataset')

    plt.savefig(f'digits_image_with_the_index_{i}_in_the_scikit-learn_digits_image_dataset.png', dpi=100)
    plt.show()


# ///////////////////////////////////////////////////////////////////////


# Using the Train Test Split Machine Learning (ML) Model Evaluation Technique to divide the dataset into two 
# seperate subsets: the training dataset subset and the test dataset subset (refer to the 'Tutorial 12 - Train 
# Test Split Machine Learning Model Evaluation Technique (Machine Learning Technique)' folder for more on the 
# Train Test Split Machine Learning (ML) Model Evaluation Technique and explanation on the code).

# Doing the Train Test Split Machine Learning (ML) Model Evaluation Technique using Scikit-learn's 'train_test_split' 
# Instance Method in the 'sklearn.model_selection' class
independent_variables = digits_image_dataset.data
dependent_variable = digits_image_dataset.target
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
# Output: 0.95

print(f'     Digit Category\n{dependent_variable_test_dataset_subset}')


# //////////////////////////////////////////////////////////////////////////////////


# Testing random sets of information in the digits image dataset with the MNLR ML model:
# Testing if the predicted 'y' nominal categorical dependent variable by the MNLR ML model is the same as the
# actual 'y' nominal categorical dependent variable

# The visualisation of the set of information in the digits image dataset at index 67 with the 'image' 
# attribute
plt.matshow(digits_image_dataset.images[67])
plt.title('Digits image with the index 67 in the Scikit-learn Digits image dataset')

plt.savefig('digits_image_with_the_index_67_in_the_scikit-learn_digits_image_dataset.png', dpi=100)
plt.show()

# The actual 'y' nominal categorical dependent variable of the set of information in the digits image 
# dataset at index 67 with the 'target' attribute
print(digits_image_dataset.target[67])            # output: 6

# # The predicted 'y' nominal categorical dependent variable by the MNLR ML model of the set of information 
# in the digits image dataset at index 67
print(multinomial_logistic_regression_ML_model.predict([digits_image_dataset.data[67]]))             # output: 6


# //////////////////////////////////////////////////////////////////////////////////


# Visualising the Confusion Matrix of the MNLR ML model with the digits image dataset:

# What is the Confusion Matrix?
# The Confusion Matrix is a performance evaluation tool used to assess the accuracy of (mostly) Classification
# Supervised Learning ML algorithms (such as Binary Logistic Regression (BLR), Multinomial Logsitic Regression 
# (MNLR), etc.) by comparing the actual 'y' categorical dependent variable with the predicted 'y' categorical 
# dependent variable by the ML model. 

# How does it work?
# From the above section 'Testing random sets of information in the digits image dataset with the MNLR ML 
# model', you can see that on those random sets of information in the digits image dataset that is being 
# used to test the MNLR ML model all show that the MNLR ML model has been doing pretty good in predicting 
# the categories correctly.

# But, from the above section 'Evaluating the trained MNLR ML model using the '.score()' Instance Method 
# of the 'Multinomial Logistic Regression (MNLR) ML model' class', you can see that the score is 0.95, 
# which means that the MNLR ML model is not perfect, and has made errors when making predictions of the
# categories.

# So how can we see where exactly the MNLR ML model made errors when making predictions?
# Thats where the Confusion Matrix comes in!


# Here is how to interpret a Confusion Matrix (using this context)?
# Looking at how the confusion matrix looks like for this context/digits image dataset,
#        [[37  0  0  0  0  0  0  0  0  0]
#         [ 0 31  1  0  1  0  0  0  1  0]
#         [ 0  0 34  0  0  0  0  0  0  0]
#         [ 0  0  1 38  0  1  0  0  0  0]
#         [ 1  1  0  0 31  0  0  0  1  0]
#         [ 0  1  0  0  0 30  0  1  0  0]
#         [ 0  1  0  0  0  0 36  0  0  0]
#         [ 0  0  0  0  1  0  0 38  0  1]
#         [ 0  0  1  0  0  0  1  0 31  0]
#         [ 0  0  0  0  0  1  0  0  2 36]]

# Each row in the confusion matrix represents the actual 'y' categorical dependent variable category of 
# the digit (0 to 9) (where each row represents the digit 0 category to the digit 9 category in order), 
# and each column represents the predicted 'y' categorical dependent variable category of the digit 
# (0 to 9) by the MNLR ML model (where each column represents the digit 0 category to the digit 9 category 
# in order), where:
# - The cell at row 0, column 0 means that the MNLR ML model predicted 37 sets of information in the 
#   digits image dataset correctly as digit 0.

# - The cell at row 1, column 1 means that the MNLR ML model predicted 31 sets of information in the 
#   digits image dataset correctly as digit 1, but it also misclassified:
#   => 1 set of information in the digits image dataset as digit 2 instead of 0 as shown by the cell 
#      at row 1, column 2
#   => 1 set of information in the digits image dataset as digit 4 instead of 0 as shown by the cell 
#      at row 1, column 4
#   => 1 set of information in the digits image dataset as digit 8 instead of 0 as shown by the cell 
#      at row 1, column 8

# - The cell at row 9, column 9 means that the MNLR ML model predicted 36 sets of information in the 
#   digits image dataset correctly as digit 9, but it misclassified:
#   => 1 set of information in the digits image dataset as digit 5 instead of 9 as shown by the cell 
#      at row 9, column 5,
#   => 2 sets of information in the digits image dataset as digit 8 instead of 9 as shown by the cell 
#      at row 9, column 8.

# - etc.


# General Observations:
# - Diagonal cells represent correct predictions of the digit categories (i.e., the MNLR ML model predicted 
#   the correct digit category in the digits image dataset).
# - Off-diagonal cells represent misclassifications of the digit categories (i.e., the MNLR ML model 
#   predicted the wrong digit category in the digits image dataset).
# - The confusion matrix shows strong/large values in the diagonal cells, indicating the MNLR ML model 
#   performs well and made correct predictions of the digit categories most of the time, but there 
#   are some occasional misclassifications of the digit categories.

# (Note: the 'dependent_variable_test_dataset_subset' variable is storing the actual 'y' categorical dependent 
#  variable while the 'predictions' variable is storing the predicted 'y' categorical dependent variable by the 
#  MNLR ML model)
predictions = multinomial_logistic_regression_ML_model.predict(independent_variables_test_dataset_subset)
confusion_matrix_of_multinomial_logistic_regression_ML_model = confusion_matrix(dependent_variable_test_dataset_subset, predictions)

print(confusion_matrix_of_multinomial_logistic_regression_ML_model)
# output:
# [[37  0  0  0  0  0  0  0  0  0]
#  [ 0 31  1  0  1  0  0  0  1  0]
#  [ 0  0 34  0  0  0  0  0  0  0]
#  [ 0  0  1 38  0  1  0  0  0  0]
#  [ 1  1  0  0 31  0  0  0  1  0]
#  [ 0  1  0  0  0 30  0  1  0  0]
#  [ 0  1  0  0  0  0 36  0  0  0]
#  [ 0  0  0  0  1  0  0 38  0  1]
#  [ 0  0  1  0  0  0  1  0 31  0]
#  [ 0  0  0  0  0  1  0  0  2 36]]



# Representing the Confusion Matrix in a more presentable manner using the Seaborn (Python) library's 
# 'heatmap()' function and Matplotlib (Python) library

# The 'plt.figure()' Matplotlib (Python) library function is optional, it just defines the dimensions/size
# of the plot (which the heatmap plot is defined with dimensions 10 inches by 7 inches in this context)
plt.figure(figsize = (10,7))
plt.title('Confusion Matrix for Multinomial Logistic Regression Machine Learning (ML)\nmodel on Scikit-learn Digits Image Dataset')
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


plt.savefig('confusion_matrix_visualisation_of_the_Multinomial_Logistic_Regression_ML_model_on_the_scikit-learn_digits_image_dataset.png', dpi=100)
plt.show()