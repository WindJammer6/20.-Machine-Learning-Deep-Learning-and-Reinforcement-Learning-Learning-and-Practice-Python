# Question 1:
# Download the employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics, 
# and save it as a 'employee_retain_or_employee_not_retain_with_multiple_independent_variables_and_
# binary_categorical_dependent_variable.csv' file.

# 1. Do some exploratory data analysis to figure out which independent variables/features have direct
#    and clear impact on employee retention (i.e. whether they leave the company or continue to work)
# 2. Plot bar charts showing impact of employee salaries on employee retention
# 3. Plot bar charts showing corelation between employee department and employee retention
# 4. Build Binary Logistic Regression (BLR) Machine Learning (ML) model using independent 
#    variables/features that were narrowed down in step 1, 2 and 3
# 5. Measure the accuracy of the BLR ML model


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Creating the visualisation of the dataset used for the Binary Logistic Regression (BLR) ML model on a graph:

# Reading the 'employee_retain_or_employee_not_retain_with_multiple_independent_variables_and_binary_categorical_
# dependent_variable.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("employee_retain_or_employee_not_retain_with_multiple_independent_variables_and_binary_categorical_dependent_variable.csv")
print(dataset)


# /////////////////////////////////////////////////////////////////////////////


# 1. Do some exploratory data analysis to figure out which independent variables/features have direct
#    and clear impact on employee retention (i.e. whether they leave the company or continue to work)
#    (Answer taken from Excercise solution, since I could not figure out how to do this...)

# The '.groupby()' function groups the rows of the dataset by the unique values (which are 0 and 1 in this
# 'employee_retain_or_employee_not_retain_with_multiple_independent_variables_and_binary_categorical_
# dependent_variable.csv' dataset) in the 'left' column.

# However, after grouping, it computes the mean of each numerical column (which are the 'satisfaction_level', 
# 'last_evaluation number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident' and 
# 'promotion_last_5years' columns for this 'employee_retain_or_employee_not_retain_with_multiple_independent_
# variables_and_binary_categorical_dependent_variableles.csv' dataset) for each group, but will ignore 
# non-numeric columns (which are the 'Department' and 'salary' columns for this 'employee_retain_or_employee_
# not_retain_with_multiple_independent_variables_and_binary_categorical_dependent_variableles.csv' dataset) 
# during the mean calculation.
print(dataset.groupby('left').mean())
# Output:
#      satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident	promotion_last_5years
# left							
#   0	    0.666810	     0.715473	    3.786664	     199.060203	          3.380032	        0.175009       	  0.026251
#   1	    0.440098	     0.718113	    3.855503	     207.419210	          3.876505	        0.047326	      0.005321


# From above table we can draw the following conclusions that for the numerical columns,
# - Satisfaction Level: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs 
#   the retained ones (0.66)
# - Average Monthly Hours: Average monthly hours are higher in employees leaving the firm (199 vs 207)
# - Promotion Last 5 Years: Employees who are given promotion are likely to be retained at firm


# /////////////////////////////////////////////////////////////////////////////


# 2. Plotting bar charts showing impact of employee salaries (non-numerical column) on employee retention
temp_dict = {}

for index, row in dataset.iterrows():
    if row['salary'] not in temp_dict:
        if row['left'] == 1.0:
            temp_dict[row['salary']] = [0,1]
        else:                                   # if 'row['left'] == 0:'
            temp_dict[row['salary']] = [1,0]
    
    else:
        if row['left'] == 1.0:
            temp_dict[row['salary']][0] += 1
        else:                                   # if 'row['left'] == 0:'
            temp_dict[row['salary']][1] += 1

print(temp_dict)

employee_not_retained_for_each_salary_type = []
for i in temp_dict:
    employee_not_retained_for_each_salary_type.append(temp_dict[i][0])
print(employee_not_retained_for_each_salary_type)

employee_retained_for_each_salary_type = []
for i in temp_dict:
    employee_retained_for_each_salary_type.append(temp_dict[i][1])
print(employee_retained_for_each_salary_type)


main_categories = []
for i in temp_dict:
    main_categories.append(i)
print(main_categories)

# Width of each bar
bar_width = 0.35  

# X positions for the main categories
x = np.arange(len(main_categories))

plt.title("Matplotlib graphical visualisation of bar charts showing impact\nof Employee Salaries on Employee Retention\nfor Excercise 1")
plt.xlabel('salary')
plt.ylabel('number of employees')
plt.xticks(x, main_categories)  # Replace x values with the names of the main categories

# Width of each bar
bar_width = 0.35  

plt.bar(x - bar_width/2, employee_not_retained_for_each_salary_type, bar_width, label='Employee not retained (represented by value of 1)', color='b')
plt.bar(x + bar_width/2, employee_retained_for_each_salary_type, bar_width, label='Employee retained (represented by value of 0)', color='r')
plt.legend()

plt.savefig('matplotlib_visualisation_of_bar_charts_showing_impact_of_employee_salaries_on_employee_retention_for_excercise_1.png', dpi=100)
plt.show()

# My interpretation:
# The impact of employee salaries (non-numerical column) on employee retention, shown from the bar chart drawn 
# is that employees with higher salaries are more likely to not leave the company than employees with 
# lower salaries.



# Excercise solution for part 2 (it does the exact same thing I did, but in 1 line of code...)
pd.crosstab(dataset.salary,dataset.left).plot(kind='bar')
plt.title("Matplotlib graphical visualisation of bar charts showing impact\nof Employee Salaries on Employee Retention\nfor Excercise 1 Solution")
plt.savefig('matplotlib_visualisation_of_bar_charts_showing_impact_of_employee_salaries_on_employee_retention_for_excercise_1_solution.png', dpi=100)
plt.show()


# /////////////////////////////////////////////////////////////////////////////


# 3. Plot bar charts showing corelation between employee department (non-numerical column) and employee retention
temp_dict = {}

for index, row in dataset.iterrows():
    if row['Department'] not in temp_dict:
        if row['left'] == 1.0:
            temp_dict[row['Department']] = [0,1]
        else:                                   # if 'row['left'] == 0:'
            temp_dict[row['Department']] = [1,0]
    
    else:
        if row['left'] == 1.0:
            temp_dict[row['Department']][0] += 1
        else:                                   # if 'row['left'] == 0:'
            temp_dict[row['Department']][1] += 1

print(temp_dict)

employee_not_retained_for_each_department_type = []
for i in temp_dict:
    employee_not_retained_for_each_department_type.append(temp_dict[i][0])
print(employee_not_retained_for_each_department_type)

employee_retained_for_each_department_type = []
for i in temp_dict:
    employee_retained_for_each_department_type.append(temp_dict[i][1])
print(employee_retained_for_each_department_type)


main_categories = []
for i in temp_dict:
    main_categories.append(i)
print(main_categories)

# Width of each bar
bar_width = 0.35  

# X positions for the main categories
x = np.arange(len(main_categories))

plt.title("Matplotlib graphical visualisation of bar charts showing corelation\nof Employee Department on Employee Retention\nfor Excercise 1")
plt.xlabel('department')
plt.ylabel('number of employees')
plt.xticks(x, main_categories)  # Replace x values with the names of the main categories

# Width of each bar
bar_width = 0.35  

plt.bar(x - bar_width/2, employee_not_retained_for_each_department_type, bar_width, label='Employee not retained (represented by value of 1)', color='b')
plt.bar(x + bar_width/2, employee_retained_for_each_department_type, bar_width, label='Employee retained (represented by value of 0)', color='r')
plt.legend()

plt.savefig('matplotlib_visualisation_of_bar_charts_showing_corelation_of_employee_department_on_employee_retention_for_excercise_1.png', dpi=100)
plt.show()

# My interpretation:
# The correlation of employee department (non-numerical column) on employee retention, shown from the bar 
# chart drawn is not very clear and does not seem to have a very major influence on employee retention.



# Excercise solution for part 3 (it does the exact same thing I did, but in 1 line of code...)
pd.crosstab(dataset.Department,dataset.left).plot(kind='bar')
plt.title("Matplotlib graphical visualisation of bar charts showing corelation\nof Employee Department on Employee Retention\nfor Excercise 1 Solution")
plt.savefig('matplotlib_visualisation_of_bar_charts_showing_corelation_of_employee_department_on_employee_retention_for_excercise_1_solution.png', dpi=100)
plt.show()


# ///////////////////////////////////////////////////////////////////////////////////////


# 4. Build Binary Logistic Regression (BLR) Machine Learning (ML) model using independent 
#    variables/features that were narrowed down in step 1, 2 and 3
# From step 1, the narrowed down numerical columns independent variables/features which have direct 
# and clear impact on employee retention are:
# From the numerical columns,
# - 'satisfaction_level' (YES)
# - 'last_evaluation number_project' (NO)
# - 'average_montly_hours' (YES)
# - 'time_spend_company' (NO)
# - 'Work_accident' (NO)
# - 'promotion_last_5years' (YES)

# From step 2 and 3, the narrowed down non-numerical columns independent variables/features which have 
# direct and clear impact on employee retention are:
# From the numerical columns,
# - 'salary' (YES)
# - 'Department' (NO)


# Since the 'salary' column in the dataset is an ordinal (since it stores the categorical values, 'low',
# 'medium' and 'high', which has a specific, inherent order or ranking among them) non-numerical column 
# independent categorical variable/feature, hence, we need to first use the Label Encoding Machine Learning 
# Encoding Technique to convert it to a numerical column independent categorical variable/feature (refer to the
# 'Tutorial 4 - Machine Learning Encoding Techniques, One-hot Encoding Machine Learning Encoding 
# Technique and Label Encoding Machine Learning Encoding Technique (Machine Learning Technique)' folder
# for more information on the Label Encoding Machine Learning Encoding Technique)

# Manually creating a Label Encoded dataset of ordinal categorical variables:
# (I chose to do this manually instead of doing this using the scikit-learn 'LabelEncoder' class because the
# scikit-learn 'LabelEncoder' class requires the ordinal categorical variable's values to be in the same correct
# ASCII order as the specific, inherent order or ranking among them in order to work, which will not work in 
# this case due to the 'salary column storing the categorical values, 'low', 'medium' and 'high', which 
# is not in the correct ASCII order as the specific, inherent order or ranking among them)


# Creating a mapping for the 'salary' column, which is storing the ordinal categorical variables, for each 
# ordinal categorical variables, 'low', 'medium' and 'high', to the values of 0, 1 and 2 respectively, which 
# indicates/preserves the specific, inherent order or ranking among them (since for Education level: 
# high > medium > low).
custom_mapping = {
    "low": 0,
    "medium": 1,
    "high": 2
}

# Applying the mapping for the 'salary' column, to the 'salary' column in the initial dataset
dataset['salary'] = dataset['salary'].map(custom_mapping)
print(dataset)


# ////////////////////////////////////////////////////////////////////////////////


# Using the Train Test Split Machine Learning (ML) Model Evaluation Technique to divide the dataset into two 
# seperate subsets: the training dataset subset and the test dataset subset (refer to the 'Tutorial 12 - Train 
# Test Split Machine Learning Model Evaluation Technique (Machine Learning Technique)' folder for more on the 
# Train Test Split Machine Learning (ML) Model Evaluation Technique and explanation on the code).

# Doing the Train Test Split Machine Learning (ML) Model Evaluation Technique using Scikit-learn's 'train_test_split' 
# Instance Method in the 'sklearn.model_selection' class
independent_variables = dataset[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
dependent_variable = dataset['left']
print(independent_variables)
print(dependent_variable)

independent_variables_training_dataset_subset, independent_variables_test_dataset_subset, dependent_variable_training_dataset_subset, dependent_variable_test_dataset_subset = train_test_split(independent_variables, dependent_variable, test_size=0.1, random_state=10)

print(f'The independent variables training dataset subset: \n{independent_variables_training_dataset_subset}')

print(f'The independent variables test dataset subset: \n{independent_variables_test_dataset_subset}')

print(f'The dependent variables training dataset subset: \n{dependent_variable_training_dataset_subset}')

print(f'The dependent variables test dataset subset: \n{dependent_variable_test_dataset_subset}')


# //////////////////////////////////////////////////////////////////////////////


# Creating a 'BLR ML model' class object/instance
binary_logistic_regression_ML_model = LogisticRegression()


# Training the 'BLR ML model' class object/instance
print(binary_logistic_regression_ML_model.fit(independent_variables_training_dataset_subset, dependent_variable_training_dataset_subset))


# The '.coef_' attribute of the 'BLR ML model' class shows the values of the weights/gradients/coefficients of the 
# independent variables/features 'x1', 'x2', 'x3' ... 'x?', 'm1', 'm2', 'm3' ... 'm?' respectively, 
# in the '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b' in the mathematical equation representing the 
# BLR ML algorithm, 'sigmoid(z) = 1 / (1 + 𝑒^−z)'
print(binary_logistic_regression_ML_model.coef_)                        # output: [[-3.77033300e+00  2.35589876e-03 -1.13574426e+00 -6.47342629e-01]]


# The '.intercept_' attribute of the 'MVLR ML model' class shows the value of the bias/intercept, 'b', in the 
# '𝑧 = m1*x1 + m2*x2 + m3*x3 + ... + m?x? + b' in the mathematical equation representing the BLR ML algorithm, 
# 'sigmoid(z) = 1 / (1 + 𝑒^−z)'
print(binary_logistic_regression_ML_model.intercept_)                   # output: 0.83326244


# Making predictions with the 'BLR ML model' class object/instance
print(binary_logistic_regression_ML_model.predict(independent_variables_test_dataset_subset))            # output: [0 0 0 ... 0 0 0]

# These are the 3 sets of (multiple) independent variables/features, 'x1', 'x2', 'x3', ... 'x?', input parameters in the 
# '.predict(X)' Instance Method of the 'BLR ML model' class that we are using to make 3 respective predictions using the 
# BLR ML model
print(independent_variables_test_dataset_subset)


# The predicted probability of the predictions with the 'BLR ML model' class object/instance being in each category of 
# the categorical dependent variable
print(binary_logistic_regression_ML_model.predict_proba(independent_variables_test_dataset_subset))


# //////////////////////////////////////////////////////////////////////////


# 5. Measure the accuracy of the BLR ML model

# Continue using the Train Test Split Machine Learning (ML) Model Evaluation Technique to evaluate the 
# performance of the trained BLR ML model (refer to the 'Tutorial 12 - Train Test Split Machine Learning Model 
# Evaluation Technique (Machine Learning Technique)' folder for more on the Train Test Split Machine Learning 
# (ML) Model Evaluation Technique and explanation on the code).

# Evaluating the trained BLR ML model using the '.score()' Instance Method of the 'Binary Logistic Regression 
# (BLR) ML model' class, using the test dataset subset obtained from the Train Test Split ML Model Evaluation Technique
print(binary_logistic_regression_ML_model.score(independent_variables_test_dataset_subset, dependent_variable_test_dataset_subset))
# Output: 0.7813333333333333

print(f'        left\n{dependent_variable_test_dataset_subset}')



# My answers are all correct except for part 1, where I did not know how to do and had to refer to the
# solution for answer.