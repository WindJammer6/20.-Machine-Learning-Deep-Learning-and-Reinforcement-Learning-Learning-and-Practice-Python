import pandas as pd

# Manually creating a Label Encoded dataset of ordinal categorical variables

# Reading the 'salary_with_ordinal_categorical_variables.csv' dataset from the CSV file
dataset = pd.read_csv("salary_with_ordinal_categorical_variables.csv")
print(dataset)


# Creating a mapping for the 'education_level' column, which is storing the ordinal categorical variables, for each 
# ordinal categorical variable, 'graduate', 'master's' and 'PhD', to the values of 0, 1 and 2 respectively, which 
# indicates/preserves the specific, inherent order or ranking among them (since for Education level: 
# PhD > master's > graduate).
# (see the file '7. creating_a_Label_Encoded_dataset_of_ordinal_categorical_variables_using_scikit-learn_
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
