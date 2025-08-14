import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Creating the visualisation of the dataset used for the Single Variable Linear Regression (SVLR) ML model on a graph:

# Reading the 'house_prices_with_single_independent_variable.csv' dataset from the CSV file using the Pandas library
dataset = pd.read_csv("house_prices_with_single_independent_variable.csv")
print(dataset)

# Plotting the (scatter) graph using the Matplotlib library
plt.xlabel('area of house/m^2')
plt.ylabel('price of house/$')
plt.scatter(dataset['area'], dataset['price'], color='red', marker='+')

plt.show()


# ///////////////////////////////////////////////////////////////////////////////////////


# Creating a 'Single Variable Linear Regression (SVLR) ML model' class object/instance
single_variable_linear_regression_ML_model = LinearRegression()


# Training the 'SVLR ML model' class object/instance
print(single_variable_linear_regression_ML_model.fit(dataset[['area']], dataset['price']))


# The '.coef_' attribute of the 'SVLR ML model' class shows the value of the gradient/coefficient of 'x', 'm', in the 
# mathematical equation representing the SVLR ML algorithm, 'y = mx + b'      
print(single_variable_linear_regression_ML_model.coef_)                        # output: [135.78767123]


# The '.intercept_' attribute of the 'SVLR ML model' class shows the value of the y-intercept, 'b', in the mathematical 
# equation representing the SVLR ML algorithm, 'y = mx + b'   
print(single_variable_linear_regression_ML_model.intercept_)                   # output: 180616.43835616432


# Making predictions with the 'SVLR ML model' class object/instance
print(single_variable_linear_regression_ML_model.predict([[3300]]))            # output: [628715.75342466]


# ////////////////////////////////////////////////////////////////////////////////


# Now that we have our 'Single Variable Linear Regression (SVLR) ML model', we can now predict the prices of house areas
# with it. 

# We have a seperate CSV file, 
# 'house_areas_without_prediction_of_their_prices_using_the_Single_Variable_Linear_Regression_ML_model.csv', which
# contains a dataset, of just a column of house areas (without the corresponding prices), we will first:
# 1. Read this dataset of house areas from the CSV file using Pandas 
# 2. Use the 'SVLR ML model' to make predictions of the prices of each of the house areas and storing a Python List of 
#    predictions for each house area into a variable
# 3. Appending a new column to the dataset with the corresponding price for each house area in this dataset
# 4. Save this new, modified dataset with the price predictions into another CSV file, 
#    'house_areas_with_prediction_of_their_prices_using_the_Single_Variable_Linear_Regression_ML_model.csv'

# 1. Read the 'house_areas_without_prediction_of_their_prices_using_the_Single_Variable_Linear_Regression_ML_model.csv' 
#    dataset of house areas from the CSV file using Pandas
dataset2 = pd.read_csv('house_areas_without_prediction_of_their_prices_using_the_Single_Variable_Linear_Regression_ML_model.csv')
print(dataset2)

# 2. Use the 'SVLR ML model' to make a Python List of predictions of the prices of each of the house areas and storing 
#    a Python List of predictions for each house area into a variable
predictions_single_variable_linear_regression_ML_model = single_variable_linear_regression_ML_model.predict(dataset2)

# 3. Appending a new column to the dataset with the corresponding price for each house area in this dataset
dataset2['prices'] = predictions_single_variable_linear_regression_ML_model

# 4. Save this new, modified dataset with the price predictions into another CSV file, 
#    'house_areas_with_prediction_of_their_prices_using_the_Single_Variable_Linear_Regression_ML_model.csv'
dataset2.to_csv('house_areas_with_prediction_of_their_prices_using_the_Single_Variable_Linear_Regression_ML_model.csv', index=False)