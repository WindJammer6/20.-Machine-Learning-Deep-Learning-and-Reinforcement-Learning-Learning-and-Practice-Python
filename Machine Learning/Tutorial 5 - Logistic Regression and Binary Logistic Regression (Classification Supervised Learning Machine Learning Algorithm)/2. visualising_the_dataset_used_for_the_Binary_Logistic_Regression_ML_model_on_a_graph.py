# In this file, we will be first visualising the dataset used for the Binary Logistic Regression (BLR)
# Supervised learning Regression Machine Learning (ML) algorithm in a graph in a context, to better understand how 
# the Binary Logistic Regression (BLR) algorithm works. 

# Note: The title of this file is labelled '2. visualising_the_dataset_used_for_the_Binary_Logistic_Regression_
#       ML_model_on_a_graph' rather than '2. visualising_the_dataset_used_for_the_Binary_Logistic_Regression_ML_
#       algorithm_on_a_graph' because this file is the visualisation of the BLR ML algorithm in a graph in a context, 
#       and since a ML model is basically a ML algorithm that is contextualised, it is more accurate to describe the 
#       code in this file as a 'Binary Logistic Regression ML model' visualised on a graph, rather than a 'Binary
#       Logistic Regression ML algorithm' visualised on a graph.

import pandas as pd
import matplotlib.pyplot as plt

# Creating the visualisation of the dataset used for the Binary Logistic Regression (BLR) ML model on a graph:

# Reading the 'bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_
# dependent_variable.csv' dataset from the CSV file 
# using the Pandas library
dataset = pd.read_csv("bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_dependent_variable.csv")
print(dataset)

# Plotting the (scatter) graph using the Matplotlib library
plt.xlabel('age')
plt.ylabel('bought insurance or not bought insurance')
plt.scatter(dataset['age'], dataset['bought_insurance'], color='red', marker='+')

plt.title("Matplotlib graphical visualisation of the dataset\nused for the Binary Logistic Regression (BLR)\nMachine Learning (ML) model")
plt.savefig('matplotlib_visualisation_of_the_dataset_used_for_the_Binary_Logistic_Regression_ML_model.png', dpi=100)
plt.show()