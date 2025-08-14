import pandas as pd
import matplotlib.pyplot as plt
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


# We will later continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to 
# evaluate the performance of the BLR ML model once it is trained.


