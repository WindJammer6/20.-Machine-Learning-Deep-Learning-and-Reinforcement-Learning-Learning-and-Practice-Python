# In the previous file: '2. General_Gradient_Descent_Optimization_Algorithm_with_Mean_Squared_Error_Cost_Function_on_
# Single_Variable_Linear_Regression_model.py', the (general) Gradient Descent Optimization Algorithm is not yet at minima. 
# This is because the value of the Cost Function is still changing by a rather high amount between iterations 
# (approximately 0.0001). Usually we will only consider the (general) Gradient Descent Optimization Algorithm to have 
# reached minima only if the value of the Cost Function changes by a threshold negligible amount 
# (e.g. 0.000000000000000000000000000000001) between iterations then it is considered to have reached minima. 

# Output from the code in the previous file:  '2. General_Gradient_Descent_Optimization_Algorithm_with_Mean_Squared_Error_
# Cost_Function_on_Single_Variable_Linear_Regression_model.py':
    # Current iteration: 0, Current weight ('m') value: 0, Current biases ('b') value: 0, Cost Function value: 89.0
    # Current iteration: 1, Current weight ('m') value: 0.062, Current biases ('b') value: 0.018000000000000002, Cost Function value: 84.881304
    # Current iteration: 2, Current weight ('m') value: 0.122528, Current biases ('b') value: 0.035592000000000006, Cost Function value: 80.955185108544
    # Current iteration: 3, Current weight ('m') value: 0.181618832, Current biases ('b') value: 0.052785648000000004, Cost Function value: 77.21263768455901

    # ...

    # Current iteration: 997, Current weight ('m') value: 2.449155141761153, Current biases ('b') value: 1.3784074216500761, Cost Function value: 0.4786263787892881
    # Current iteration: 998, Current weight ('m') value: 2.449003284112507, Current biases ('b') value: 1.3789556759562092, Cost Function value: 0.4783027899709678
    # Current iteration: 999, Current weight ('m') value: 2.448851477806295, Current biases ('b') value: 1.3795037448996217, Cost Function value: 0.47797941992396514


import numpy as np


# What this function does is that it implements the (general) Gradient Descent Optimization Algorithm with the
# Mean Squared Error (MSE) Cost Function, in context of the Single Variable Linear Regression (SVLR) ML algorithm.
def general_gradient_descent_optimization_algorithm(x, y):

    current_weight_m_value = 0
    current_biases_b_value = 0



    # In order to get the (general) Gradient Descent Optimization Algorithm to reach the minima, there is 2 parameters/values we
    # can further change, by trial and error, in order to get the set of values of the current weight ('m') and current biases ('b') 
    # at the final iteration where the value of the Cost Function changes below the threshold negligible amount 
    # (e.g. 0.000000000000000000000000000000001), which represents the line/the SVLR ML algorithm with the optimal/minimial value of 
    # the Cost Function:
    # - 'number_of_iterations'
    # - 'learning_rate_value'

    # The general method to change these 2 parameters/values by trial and error is as such:
    # a. In the first part of this method, we will try to find an approximately optimal/largest possible value of the 'learning_rate', 
    #    such that the (general) Gradient Descent Optimization Algorithm just barely not miss the minimma. 
    #    1. Start with a low value of the 'number_of_iterations' (about 10)
    #    2. Use some random value of the 'learning_rate' (about 0.001)
    #    3. Keep trying other random values (e.g. 0.01, 0.05, 0.1, etc.), until you find a value of the 'learning_rate' that just 
    #       barely not miss the minima. (e.g. in this case, the 'learning_rate' value of 0.09 will miss the minima, but the 
    #       'learning_rate' value of 0.08 does not miss the minima, hence in this case, the 'learning_rate' value of 0.08 is the 
    #       'learning_rate' value that just barely not miss the minima).

    # b. In the second of this method, we will then find the appropriate value of the 'number_of_iterations', such that there is 
    #    sufficient 'number_of_iterations' for the the (general) Gradient Descent Optimization Algorithm to get the set of values of 
    #    the current weight ('m') and current biases ('b') at the final iteration where the value of the Cost Function changes below 
    #    the threshold negligible amount (e.g. 0.000000000000000000000000000000001).
    #    4. Try different values of the 'number_of_iterations' until you find the value of the Cost Function changes by a threshold 
    #       negligible amount (e.g. 0.000000000000000000000000000000001) between iterations within that 'number_of_iterations' before 
    #       the end of these 'number_of_iterations'. (e.g. 100, 1000, 10000)
    #    5. Set a limit of the threshold negligible amount (e.g. 0.000000000000000000000000000000001) of the value changed by the
    #       Cost Function between iterations by telling the 'general_gradient_descent_optimization_algorithm()''s code to break
    #       out of the for loop earlier than the initiated 'number_of_iterations' if the the value changed by the Cost Function between 
    #       iterations is less than the limit threshold negligible amount (e.g. 0.000000000000000000000000000000001) (this step 5
    #       is done in the code further below)

    # This general method to change these 2 parameters/values by trial and error, by first finding the approximately optimal/largest
    # possible value of the 'learning_rate', such that the (general) Gradient Descent Optimization Algorithm just barely not miss the 
    # minimma, then secondly, finding the appropriate value of the 'number_of_iterations' is recommended as it allows the (general)
    # Gradient Descent Optimization Algorithm to reduce the 'number_of_iterations' required to get the set of values of the 
    # current weight ('m') and current biases ('b') at the final iteration where the value of the Cost Function changes below the 
    # threshold negligible amount (e.g. 0.000000000000000000000000000000001) between iterations.

    # In this case, the value of the 'learning_rate' of 0.08, the approximately optimal/largest possible value of the 'learning_rate', 
    # will require only about 1157 'number_of_iterations' compared to the value of the 'learning_rate' of 0.01, which will require
    # about 8669 'number_of_iterations' for the (general) Gradient Descent Optimization Algorithm to get the set of values of the current 
    # weight ('m') and current biases ('b') at the final iteration where the value of the Cost Function changes below the threshold 
    # negligible amount (e.g. 0.000000000000000000000000000000001)!
    number_of_iterations = 10000
    n = len(x)
    learning_rate_value = 0.08



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
        



        # Code of step 5 (within part b) of the general method to change these 2 parameters/values by trial and error

        #    5. Set a limit of the threshold negligible amount (e.g. 0.000000000000000000000000000000001) of the value changed by the
        #       Cost Function between iterations by telling the 'general_gradient_descent_optimization_algorithm()''s code to break
        #       out of the for loop earlier than the initiated 'number_of_iterations' if the the value changed by the Cost Function between 
        #       iterations is less than the limit threshold negligible amount (e.g. 0.000000000000000000000000000000001) (this step 5
        #       is done in the code further below)

        # (Note that this code of step 5 (within part b) of the general method to change these 2 parameters/values by trial and error will
        #  create an error if you selected a value of the 'learning_rate' that causes the (general) Gradient Descent Optimization Algorithm
        #  to miss the minima)
        if current_cost_function_value - ((1/n) * sum(((y - (current_weight_m_value * x + current_biases_b_value)))**2)) < 0.000000000000000000000000000000001:
            break
        else:
            pass



if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    general_gradient_descent_optimization_algorithm(x, y)


# /////////////////////////////////////////////////////////////////////////////////////////////////


# And we will take the set of values of the current weight ('m') and current biases ('b') at the final iteration where the value 
# of the Cost Function changes below the threshold negligible amount (e.g. 0.000000000000000000000000000000001) between 
# iterations as the founded set of values of the weights ('m') and biases ('b') of the line/the SVLR ML algorithm with the 
# optimal/minimial value of the Cost Function.

# Compared to from the previous file: '2. General_Gradient_Descent_Optimization_Algorithm_with_Mean_Squared_Error_Cost_Function_on_
# Single_Variable_Linear_Regression_model.py' the (general) Gradient Descent Optimization Algorithm did not yet reach minima. This 
# is because the value of the Cost Function is still changing by a rather high amount between iterations (approximately 0.0001). 
# Usually we will only consider the (general) Gradient Descent Optimization Algorithm to have reached minima only if the value of 
# the Cost Function changes by a threshold negligible amount (e.g. 0.000000000000000000000000000000001) between iterations then it 
# is considered to have reached minima. 

# Now in this file, after further changing the 2 parameters/values ('number_of_iterations' and 'learning_rate_value'), by trial and 
# error, we have reached minima, due to at the final iteration of the (general) Gradient Descent Optimization Algorithm in this file
# having a set of values of the current weight ('m') and current biases ('b') where the value of the Cost Function changes below the 
# threshold negligible amount (e.g. 0.000000000000000000000000000000001), which represents the line/the SVLR ML algorithm with the 
# optimal/minimial value of the Cost Function.

# In this case, this set of value the current weight ('m') and current biases ('b') where the value of the Cost Function changes below 
# the threshold negligible amount (e.g. 0.000000000000000000000000000000001), which represents the line/the SVLR ML algorithm with the 
# optimal/minimial value of the Cost Function, are 
        # Current weight ('m') value: 2.00000000000001  |   Current biases ('b') value: 2.999999999999963 

# Hence, it can be said that using the (general) Gradient Descent Optimization Algorithm with the Mean Squared Error (MSE) Cost 
# Function, we have successfully 'trained' our SVLR ML algorithm into a SVLR ML model using the provided dataset (which is 
# x = np.array([1, 2, 3, 4, 5])  |  y = np.array([5, 7, 9, 11, 13]))), when we found this set of value the current weight ('m') and 
# current biases ('b') for the best fit line/SVLR ML model (represented by the mathematical formula/equation 'y = mx + b') for 
# our dataset.

# Output:
    # Current iteration: 0, Current weight ('m') value: 0, Current biases ('b') value: 0, Cost Function value: 89.0
    # Current iteration: 1, Current weight ('m') value: 4.96, Current biases ('b') value: 1.44, Cost Function value: 71.10560000000002
    # Current iteration: 2, Current weight ('m') value: 0.4991999999999983, Current biases ('b') value: 0.26879999999999993, Cost Function value: 56.8297702400001
    # Current iteration: 3, Current weight ('m') value: 4.451584000000002, Current biases ('b') value: 1.426176000000001, Cost Function value: 45.43965675929613
    # ...
    # Current iteration: 1155, Current weight ('m') value: 2.000000000000011, Current biases ('b') value: 2.999999999999961, Cost Function value: 2.8099225443972444e-28
    # Current iteration: 1156, Current weight ('m') value: 2.00000000000001, Current biases ('b') value: 2.999999999999962, Cost Function value: 2.7767903863779616e-28
    # Current iteration: 1157, Current weight ('m') value: 2.00000000000001, Current biases ('b') value: 2.999999999999963, Cost Function value: 2.461246024289557e-28