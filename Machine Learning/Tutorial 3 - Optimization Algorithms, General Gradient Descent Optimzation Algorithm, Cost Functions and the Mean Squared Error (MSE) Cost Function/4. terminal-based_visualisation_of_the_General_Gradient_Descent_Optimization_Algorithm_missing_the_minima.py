# In the previous file: '3. changing_the_learning_rate_value_and_number_of_iterations_value_to_find_the_set_of_weights_and_
# biases_with_the_minimal_Cost_Function_value_in_the_General_Gradient_Descent_Optimization_Algorithm.py', we mentioned that
# if the value of the 'learning_rate' is too large, it might cause the (general) Gradient Descent Optimization Algorithm
# to miss the minima.


# But how exactly does 'missing the minima' look like in the output, if the value of the 'learning_rate' is too large?
# In the previous file: '3. changing_the_learning_rate_value_and_number_of_iterations_value_to_find_the_set_of_weights_and_
# biases_with_the_minimal_Cost_Function_value_in_the_General_Gradient_Descent_Optimization_Algorithm.py', through trial
# and error, we found that in this case, the 'learning_rate' value of 0.09 and larger will miss the minima, but the 
# 'learning_rate' value of 0.08  just barely not miss the minima.

# Hence, in this file, we will intentionally use a value of the 'learning_rate' that is too large so that the (general)
# Gradient Descent Optimization Algorithm will miss the minima, and run the code to see what the output is for when the 
# (general) Gradient Descent Optimization Algorithm misses the minima.


# ////////////////////////////////////////////////////////////////////////////////////////////


import numpy as np

# What this function does is that it implements the (general) Gradient Descent Optimization Algorithm with the
# Mean Squared Error (MSE) Cost Function, in context of the Single Variable Linear Regression (SVLR) ML algorithm.
def general_gradient_descent_optimization_algorithm(x, y):

    current_weight_m_value = 0
    current_biases_b_value = 0

    number_of_iterations = 10000
    n = len(x)
    learning_rate_value = 0.09

    for i in range(number_of_iterations):

        y_predicted = current_weight_m_value * x + current_biases_b_value

        current_cost_function_value = (1/n) * sum(((y - (current_weight_m_value * x + current_biases_b_value)))**2)
        # current_cost_function_value = (1/n) * sum(((y - y_predicted))**2) works too

        partial_derivative_of_weight_m_with_respect_to_Cost_Function = -(2/n) * sum(x * (y - (current_weight_m_value * x + current_biases_b_value)))
        # partial_derivative_of_weight_m_with_respect_to_Cost_Function = -(2/n) * sum(x*(y - y_predicted)) works too 
        partial_derivative_of_biases_b_with_respect_to_Cost_Function = -(2/n) * sum((y - (current_weight_m_value * x + current_biases_b_value)))
        # partial_derivative_of_biases_b_with_respect_to_Cost_Function = -(2/n) * sum(x*(y - y_predicted)) works too

        print("Current iteration: {}, Current weight ('m') value: {}, Current biases ('b') value: {}, Cost Function value: {}".format(i, current_weight_m_value, current_biases_b_value, current_cost_function_value))
        
        current_weight_m_value = current_weight_m_value - (learning_rate_value * partial_derivative_of_weight_m_with_respect_to_Cost_Function)
        current_biases_b_value = current_biases_b_value - (learning_rate_value * partial_derivative_of_biases_b_with_respect_to_Cost_Function)



        # Code of step 5 (within part b) of the general method to change these 2 parameters/values by trial and error (see what this 
        # 'general method' is in the previous file '3. changing_the_learning_rate_value_and_number_of_iterations_value_to_find_the_set_
        # of_weights_and_biases_with_the_minimal_Cost_Function_value_in_the_General_Gradient_Descent_Optimization_Algorithm)

        #    5. Set a limit of the threshold negligible amount (e.g. 0.000000000000000000000000000000001) of the value changed by the
        #       Cost Function between iterations by telling the 'general_gradient_descent_optimization_algorithm()''s code to break
        #       out of the for loop earlier than the initiated 'number_of_iterations' if the the value changed by the Cost Function between 
        #       iterations is less than the limit threshold negligible amount (e.g. 0.000000000000000000000000000000001) (this step 5
        #       is done in the code further below)

        # This code of step 5 (within part b) of the general method to change these 2 parameters/values by trial and error will create an 
        # error if you selected a value of the 'learning_rate' that causes the (general) Gradient Descent Optimization Algorithm to miss 
        # the minima, hence in this file, this code is commented out.

        # if current_cost_function_value - ((1/n) * sum(((y - (current_weight_m_value * x + current_biases_b_value)))**2)) < 0.000000000000000000000000000000001:
        #     break
        # else:
        #     pass



if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    general_gradient_descent_optimization_algorithm(x, y)


# /////////////////////////////////////////////////////////////////////////////////////////////////


# But how exactly does 'missing the minima' look like in the output, if the value of the 'learning_rate' is too large?
# Usually, when the appropriate value of the 'learning_rate' is chosen, such that the (general) Gradient Descent Optimization 
# Algorithm does not miss the minima, the value of the Cost Function will keep decreasing throughout the iterations.

# However, when an inappropriate (too large) value of the 'learning_rate' is chosen such as the case in this file, such that the 
# (general) Gradient Descent Optimization Algorithm misses the minima, the value of the Cost Function will keep increasing, rather 
# than decrease throughout the iterations, as shown from the output of this file.

# Output (of the (general) Gradient Descent Optimization Algorithm 'missing' the minima) of this file:
    # Current iteration: 0, Current weight ('m') value: 0, Current biases ('b') value: 0, Cost Function value: 89.0
    # Current iteration: 1, Current weight ('m') value: 4.96, Current biases ('b') value: 1.44, Cost Function value: 71.10560000000002
    # Current iteration: 2, Current weight ('m') value: 0.4991999999999983, Current biases ('b') value: 0.26879999999999993, Cost Function value: 56.8297702400001
    # Current iteration: 3, Current weight ('m') value: 4.451584000000002, Current biases ('b') value: 1.426176000000001, Cost Function value: 45.43965675929613
    # ...
    # Current iteration: 1155, Current weight ('m') value: 2.000000000000011, Current biases ('b') value: 2.999999999999961, Cost Function value: 2.8099225443972444e-28
    # Current iteration: 1156, Current weight ('m') value: 2.00000000000001, Current biases ('b') value: 2.999999999999962, Cost Function value: 2.7767903863779616e-28
    # Current iteration: 1157, Current weight ('m') value: 2.00000000000001, Current biases ('b') value: 2.999999999999963, Cost Function value: 2.461246024289557e-28
    