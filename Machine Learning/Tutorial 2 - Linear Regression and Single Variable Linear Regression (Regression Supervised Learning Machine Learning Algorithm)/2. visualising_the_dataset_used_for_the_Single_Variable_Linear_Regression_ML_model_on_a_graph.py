# In this file, we will be first visualising the dataset used for the Single Variable Linear Regression (SVLR) 
# Supervised learning Regression Machine Learning (ML) algorithm in a graph in a context, to better understand how 
# the Single Variable Linear Regression algorithm works. 

# Note: The title of this file is labelled '2. visualising_the_dataset_used_for_the_Single_Variable_Linear_Regression_
#       ML_model_on_a_graph' rather than '2. visualising_the_dataset_used_for_the_Single_Variable_Linear_Regression_ML_
#       algorithm_on_a_graph' because this file is the visualisation of the SVLR ML algorithm in a graph in a context, 
#       and since a ML model is basically a ML algorithm that is contextualised, it is more accurate to describe the 
#       code in this file as a 'Single Variable Linear Regression ML model' visualised on a graph, rather than a 'Single 
#       Variable Linear Regression ML algorithm' visualised on a graph.

import pandas as pd
import matplotlib.pyplot as plt

# Creating the visualisation of the dataset used for the Single Variable Linear Regression (SVLR) ML model on a graph:

# Reading the 'house_prices_with_single_independent_variable.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("house_prices_with_single_independent_variable.csv")
print(dataset)

# Plotting the (scatter) graph using the Matplotlib library
plt.xlabel('area of house/m^2')
plt.ylabel('price of house/$')
plt.scatter(dataset['area'], dataset['price'], color='red', marker='+')

plt.title("Matplotlib graphical visualisation of the dataset\nused for the Single Variable Linear Regression (SVLR)\nMachine Learning (ML) model")
plt.savefig('matplotlib_visualisation_of_the_dataset_used_for_the_Single_Variable_Linear_Regression_ML_model.png', dpi=100)
plt.show()