import pandas as pd


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
# (If you refer to the '2. manually_creating_an_One-hot_Encoded_dataset_of_nominal_categorical_variables_
#  using_dummy_variables.py' file in the 'Tutorial 4 - Machine Learning Encoding Techniques, One-hot Encoding 
#  Machine Learning Encoding Technique and One-hot Encoding Machine Learning Encoding Technique (Machine 
#  Learning Technique)' folder, you will notice in that file, the way of manually creating an One-hot
#  Encoded dataset of nominal categorical variables is by first creating dummy variables and then dropping
#  one of the dummy variables (to avoid the 'Dummy Variable Trap'). However, in this file, I chose to instead
#  just custom map the each nominal dependent categorical variable to the corresponding values. Because 
#  manually creating an One-hot Encoded dataset of nominal categorical variables via creating dummy variables 
#  complicates the training of the MNLR ML algorithm using the Scikit-learn Machine Learning (ML) library,
#  but by just custom mapping the each nominal dependent categorical variable to the corresponding values 
#  simplify things)


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