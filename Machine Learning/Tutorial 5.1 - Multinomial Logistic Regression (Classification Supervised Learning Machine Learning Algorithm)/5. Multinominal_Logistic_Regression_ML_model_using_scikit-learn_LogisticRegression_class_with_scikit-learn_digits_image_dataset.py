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


# Reading the digits image dataset
digits_image_dataset = load_digits()
print(digits_image_dataset)

# Using the 'dir()' function to see the available attributes of the digits image dataset
# There are 7 available attributes of the digits image dataset, but the most important 3 attributes
# are:
# - 'data'   - it contains the values of the independent variables/features of the digits image 
#              dataset, where each independent variable/feature represents a pixel, and there will be 
#              64 independent variable/features in this digits image dataset due to the size of each 
#              image being 8x8 pixels
# - 'target' - it contains the nominal categorical dependent variable of the digits image dataset,
#              where each nominal category representing the digit (0 to 9) the image of pixels is 
#              supposed to be showing, and there will be 10 nominal categories of the nominal 
#              categorical dependent variable in this digits image dataset due to each 
#              category representing the digit of 0 to 9 the image of pixels is supposed to be 
#              showing
# - 'images' - it contains the actual 8x8 pixel images of the digits in the form of a 2D array of
#              shape (8,8). You can use Matplotlib (Python) library's 'matshow()' function to
#              visualise these images (convert the 2D array of shape (8,8) to an image)
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