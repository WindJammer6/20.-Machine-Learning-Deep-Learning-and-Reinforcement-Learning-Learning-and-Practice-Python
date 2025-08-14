import pandas as pd
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