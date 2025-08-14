# Recalling from the file: '1. What_are_Optimization_Algorithms_and_General_Gradient_Descent_Optimization_Algorithm_
# and_what_is_a_Cost_Function_and_Mean_Squared_Error_Cost_Function.txt', in the 'What are Optimization Algorithms?'
# section,
    # Optimization Algorithms are algorithms that are used to optimize/minimise the value of the Cost 
    # Function. It does so by finding a set of parameters (not the dependent variable nor the independent 
    # variables/features, but the weights and biases) for a ML algorithm/model that optimizes/minimises the 
    # value of the Cost Function.

# by doing so, these Optimization Algorithms are able to 'train' ML algorithms into ML models, which are capable of 
# predicting outcomes and classifying information without human intervention.

# In this file, we will be using Matplotlib Python library to create a graphical representation of the (general)
# Gradient Descent Optimization Algorithm with Mean Squared Error (MSE) Cost Function 'training' a Single Variable
# Linear Regression (SVLR) ML model by iteratively finding a set of parameters (weights ('m') and biases ('b')) for 
# the SVLR ML algorithm/model that optimizes/minimises the value of the Cost Function.


# /////////////////////////////////////////////////////////////////////////////////////////////


# An image of this graphical representation is in the 'matplotlib_visualisation_of_the_General_Gradient_Descent_
# Optimization_Algorithm_with_Mean_Squared_Error_Cost_Function_training_the_Single_Variable_Linear_Regression_
# model.png' file. 

# You can see from this graphical representation that there are many line plots, with many of them converging
# towards a best fit line/SVLR ML model of the dataset. The initial line plots are further away from the 
# best fit line/SVLR ML model (due to the set of parameters (weights ('m') and biases ('b')) for the SVLR ML 
# algorithm/model not being optimal/minimising the Cost Function), but after many iterations, as part of the (general) 
# Gradient Descent Optimization Algorithm with the Mean Squared Error (MSE) Cost Function, the line plots converges 
# towards a line, which is the best fit line/SVLR ML model of the dataset (due to the (general) Gradient Descent 
# Optimization Algorithm gradually finding the set of parameters (weights ('m') and biases ('b')) for the SVLR ML 
# algorithm/model that optimizes/minimises the value of the Cost Function). 

# The (not so clear due to the many line plots) red scatter plot represents the actual values/datapoints of the 
# initial/provided dataset (x = np.array([1, 2, 3, 4, 5])  |  y = np.array([5, 7, 9, 11, 13]))). 

# Hence, this graphical representation shows the process of the (general) Gradient Descent Optimization Algorithm 
# with Mean Squared Error (MSE) Cost Function 'training' a Single Variable Linear Regression (SVLR) ML model.



import numpy as np
import matplotlib.pyplot as plt


# What this function does is that it implements the (general) Gradient Descent Optimization Algorithm with the
# Mean Squared Error (MSE) Cost Function, in context of the Single Variable Linear Regression (SVLR) ML algorithm.
def general_gradient_descent_optimization_algorithm(x, y):

    current_weight_m_value = 0
    current_biases_b_value = 0

    number_of_iterations = 10000
    n = len(x)
    learning_rate_value = 0.08

    for i in range(number_of_iterations):

        y_predicted = current_weight_m_value * x + current_biases_b_value

        current_cost_function_value = (1/n) * sum(((y - (current_weight_m_value * x + current_biases_b_value)))**2)

        partial_derivative_of_weight_m_with_respect_to_Cost_Function = -(2/n) * sum(x*(y - (current_weight_m_value * x + current_biases_b_value)))
        partial_derivative_of_biases_b_with_respect_to_Cost_Function = -(2/n) * sum((y - (current_weight_m_value * x + current_biases_b_value)))

        print("Current iteration: {}, Current weight ('m') value: {}, Current biases ('b') value: {}, Cost Function value: {}".format(i, current_weight_m_value, current_biases_b_value, current_cost_function_value))
        
        current_weight_m_value = current_weight_m_value - (learning_rate_value * partial_derivative_of_weight_m_with_respect_to_Cost_Function)
        current_biases_b_value = current_biases_b_value - (learning_rate_value * partial_derivative_of_biases_b_with_respect_to_Cost_Function)



        # (Matplotlib code) Generating a line plot (represented by the mathematical formula/equation 'y = mx + b') 
        # for every iteration's 'current_weight_m_value' and 'current_biases_b_value' 
        x_coordinates = np.linspace(0, 15, 10)
        y_coordinates = current_weight_m_value * x_coordinates + current_biases_b_value
        plt.plot(x_coordinates, y_coordinates)



        if current_cost_function_value - ((1/n) * sum(((y - (current_weight_m_value * x + current_biases_b_value)))**2)) < 0.000000000000000000000000000000001:
            break
        else:
            pass



if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    general_gradient_descent_optimization_algorithm(x, y)


    # (Matplotlib code) Generating a scatter plot of the actual values/datapoints of the initial/provided dataset
    # (x = np.array([1, 2, 3, 4, 5])  |  y = np.array([5, 7, 9, 11, 13])))
    plt.scatter(x, y, color='red')
    
    # (Matplotlib code) Adding details to the Matplotlib graphical visualisation (e.g. labels, titles, setting the 
    # range of values of the axis)
    plt.title("Matplotlib graphical visualisation of the (general) Gradient Descent\nOptimization Algorithm with Mean Squared Error (MSE) Cost Function\ntraining the Single Variable Linear Regression (SVLR)\nMachine Learning (ML) model")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xlim(0, 5)  # Fixing the range of values for the x-axis
    plt.ylim(0, 15) # Fixing the range of values for the y-axis

    plt.savefig('matplotlib_visualisation_of_the_General_Gradient_Descent_Optimization_Algorithm_with_Mean_Squared_Error_Cost_Function_training_the_Single_Variable_Linear_Regression_ML_model.png', dpi=100)
    plt.show()