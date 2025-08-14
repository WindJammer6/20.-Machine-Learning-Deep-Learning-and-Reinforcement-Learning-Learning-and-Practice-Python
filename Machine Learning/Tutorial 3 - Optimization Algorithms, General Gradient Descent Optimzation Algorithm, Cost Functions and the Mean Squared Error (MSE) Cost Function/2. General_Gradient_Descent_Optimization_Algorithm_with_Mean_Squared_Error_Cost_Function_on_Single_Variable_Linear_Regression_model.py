# In this tutorial we will not be covering all types of Optimization Algorithms and Cost Functions.
# Instead we will only be foccusing on the (general) Gradient Descent Optimization Algorithm with the Mean
# Squared Error (MSE) Cost Function, in context of the Single Variable Linear Regression (SVLR) ML algorithm.
# Hence in this file, we will be implementing the (general) Gradient Descent Optimization Algorithm with the 
# Mean Squared Error (MSE) Cost Function, in context of the Single Variable Linear Regression (SVLR) ML 
# algorithm.

# You can think of this as an example of a type of Optimisation Algorithm with the Cost Function code as a 
# possible code behind the scenes of the '.fit()' Scikit-learn function, that allows it to be able to 'train' 
# ML algorithms to become ML models, which are capable of predicting outcomes and classifying information 
# without human intervention. 


import numpy as np


# What this function does is that it implements the (general) Gradient Descent Optimization Algorithm with the
# Mean Squared Error (MSE) Cost Function, in context of the Single Variable Linear Regression (SVLR) ML algorithm.


# Here are the steps of the (general) Gradient Descent Optimization Algorithm:

# 1. Imagine a 3D space, with,
#    - the x-axis representing the range of values of weights ('m')
#    - the y-axis representing the range of values of biases ('b')
#    - the z-axis representing the range of values of the Cost Function (Mean Squared Error (MSE) Cost Function in 
#       this context)

#    And a plane with a rough shape of a 'bowl' (but not exactly the symmtrical shape of a 'bowl')

#    This plane represents the plot of all possible values of the Cost Function (MSE Cost Function in this context) for
#    every combination of the set of values of the weights ('m') and biases ('b') of the line/the SVLR ML algorithm (with 
#    the mathematical equation, 'y = mx + b'), which in turns represents all the possible positions of the line/the SVLR 
#    ML algorithm through a set of points.

#    (Note: Its very hard to draw this 'bowl' shaped plane in the 3D space in text. Refer to this codebasics Youtube 
#    video, titled 'Machine Learning Tutorial Python - 4: Gradient Descent and Cost Function' 
#    (link: https://www.youtube.com/watch?v=vsWrXfO3wWw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=7), at 
#    timestamp 5:20 for the visualisation of this 'bowl' shaped plane in the 3D space)
def general_gradient_descent_optimization_algorithm(x, y):



    # 2. Select a random set of values of the weights ('m') and biases ('b') on the 'bowl' shaped plane in this 3D space. This
    #    translates to selecting a random position of the the line/the SVLR ML algorithm through the set of points.
    current_weight_m_value = 0
    current_biases_b_value = 0

    # (Not part of the (general) Gradient Descent Optimization Algorithm, but is important in coding purposes) Here are 
    # some important variables that are needed to be pre-defined outside of the for-loop in order for the (general) 
    # Gradient Descent Optimization Algorithm to work, including the 'learning_rate_value' variable. What each of
    # these variable does is further explained within the steps of the (general) Gradient Descent Optimization Algorithm 
    # below in comments.
    number_of_iterations = 1000
    n = len(x)
    learning_rate_value = 0.001



    # 3. Then, we will move the values of the weights ('m') and biases ('b') by a 'step' (which means modify the selected 
    #    value of the weights ('m') and biases ('b') by a certain amount) towards the minima (in mathematical terms: This
    #    refers to lowest point/value of a graph) of the 'bowl' shaped plane in this 3D space.
    for i in range(number_of_iterations):



        #    But how exactly do we determine the certain amount of this 'step'? 
        #    A brute-force way is to use fixed 'steps' to move the values of the weights ('m') and biases ('b') towards the 
        #    minima (in mathematical terms: This refers to lowest point/value of a graph) of the 'bowl' shaped plane in this 3D 
        #    space, like so:

        #    (Note: Its very hard to draw this visualisation in text. Refer to this codebasics Youtube 
        #    video, titled 'Machine Learning Tutorial Python - 4: Gradient Descent and Cost Function' 
        #    (link: https://www.youtube.com/watch?v=vsWrXfO3wWw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=7), at 
        #    timestamp 8:45 for the visualisation)

        #    However, this way has a risk of missing the minima (in mathematical terms: This refers to lowest point/value of a 
        #    graph) of the 'bowl' shaped plane in this 3D space, as illustrated in the visualisation above.


        #    So this step of the (general) Gradient Descent Optimization Algorithm uses a way that will work better, and will 
        #    not have the problem of potentially missing the but the minima (in mathematical terms: This refers to lowest 
        #    point/value of a graph) of the 'bowl' shaped plane in this 3D space explanation is a little mathematical as it 
        #    involves derivatives (in mathematical terms: This refers to the rate of change of a function with respect to a 
        #    variable).

        #    The way that the (general) Gradient Descent Optimization Algorithm uses is represented by the following visualisation
        #    like so:

        #    (Note: Its very hard to draw this visualisation in text. Refer to this codebasics Youtube 
        #    video, titled 'Machine Learning Tutorial Python - 4: Gradient Descent and Cost Function' 
        #    (link: https://www.youtube.com/watch?v=vsWrXfO3wWw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=7), at 
        #    timestamp 9:11 for the visualisation)  
        
        #    Notice that each 'step' follows the gradient/curvature of the 'bowl' shaped plane in this 3D space that represents 
        #    all possible values of the Cost Function (MSE Cost Function in this context) for every combination of the set of 
        #    values of the weights ('m') and biases ('b') of the line/the SVLR ML algorithm (with the mathematical equation, 
        #    'y = mx + b'). Each 'step' follows the gradient/curvature, in the sense that, the 'step' size is initially large,
        #    but gradually the 'step' size gets smaller and smaller as the values of the weights ('m') and biases ('b') move 
        #    closer and closer towards the minima (in mathematical terms: This refers to lowest point/value of a graph).
        
        #    To calculate exactly how much to reduce the 'step' size mathematically, we need to calculate the value of the 
        #    gradient at the point on the 'bowl' shaped plane in this 3D space that represents all possible values of the 
        #    Cost Function (MSE Cost Function in this context) for every combination of the set of values of the weights ('m') 
        #    and biases ('b') of the line/the SVLR ML algorithm (with the mathematical equation, 'y = mx + b'), after every 
        #    'step', and make use of derivatives and partial derivatives.

        #    I will not go into the deriving process of the maths of the derivatives and partial derivatives that allow us to
        #    exactly determine the certain amount of every 'step' in each iteration, but here are the mathematical formulas:

        #          From the Cost Function (Mean Squared Error (MSE) Cost Function in this context)'s mathematical 
        #          formula/equation, 
        #             MSE = (1/n) * Σᵢ(yᵢ - (mx + b))²
        y_predicted = current_weight_m_value * x + current_biases_b_value       # You can see this as the SVLR ML algorithm, 
                                                                                # represented by the mathematical equation 
                                                                                # 'y = mx + b'
        current_cost_function_value = (1/n) * sum(((y - (current_weight_m_value * x + current_biases_b_value)))**2)
        # current_cost_function_value = (1/n) * sum(((y - y_predicted))**2) works too


        #          Find the partial derivatives with respect to the variable 'm' (weights) and 'b' (biases) respectively,
        #             dMSE/dm = -(2/n) * Σᵢxᵢ(yᵢ - (mxᵢ + b))
        partial_derivative_of_weight_m_with_respect_to_Cost_Function = -(2/n) * sum(x * (y - (current_weight_m_value * x + current_biases_b_value)))
        # partial_derivative_of_weight_m_with_respect_to_Cost_Function = -(2/n) * sum(x*(y - y_predicted)) works too 

        #             dMSE/db = -(2/n) * Σᵢ(yᵢ - (mxᵢ + b))
        partial_derivative_of_biases_b_with_respect_to_Cost_Function = -(2/n) * sum((y - (current_weight_m_value * x + current_biases_b_value)))
        # partial_derivative_of_biases_b_with_respect_to_Cost_Function = -(2/n) * sum(x*(y - y_predicted)) works too

        #          - yᵢ: represents actual data point value
        #          - ŷᵢ = (mxᵢ + b): represents prediction/data point value 
        #          - xᵢ: represents the input / independent variable/feature that corresponds to the prediction/data point value
        #          - n: represents the number of data points



        # (Not part of the (general) Gradient Descent Optimization Algorithm, but for debugging purposes) Printing out 
        # the values of the 'current_weight_m_value', 'current_biases_b_value', 'current_cost_function_value' before 
        # updating the values of the 'current_weight_m_value' and 'current_biases_b_value' below. This is to keep 
        # track of their values during each iteration.

        # Also calculating and then printing out the value of the 'current_cost_function_value' in each iteration to
        # keep track of its values during each iteration for 2 reasons:
        # - This is crucial as it allows us to see when the ((general) Gradient Descent) Optimisation Algorithm has 
        #   found the set of values of the weights ('m') and biases ('b') where the value of the Cost Function has 
        #   reached its minima/optimal/minimal, which is the objective of all Optimisation Algorithms (not just the 
        #   (general) Gradient Descent Optimization algorithm). (We will show how to determine if the (general) 
        #   Gradient Descent Optimization Algorithm has found the set of values of the weights ('m') and biases ('b') 
        #   where the value of the Cost Function has reached its minima/optimal/minimal in the file '3. changing_the_
        #   learning_rate_value_and_number_of_iterations_value_to_find_the_set_of_weights_and_biases_with_the_minimal
        #   _Cost_Function_value_in_the_General_Gradient_Descent_Optimization_Algorithm.py', and in step 4 of the (general) 
        #   Gradient Descent Optimization Algorithm below in comments)
        # - This is also crucial as it allows us to see if the value of the 'current_cost_function_value' is still going 
        #   towards the minima or if it has missed the minima. Recall from step 3 of the (general) Gradient Descent 
        #   Optimization Algorithm, it compares step 3 of the (general) Gradient Descent Optimization Algorithm to a 
        #   brute-force way of using fixed 'steps' to move the values of the weights ('m') and biases ('b') towards the 
        #   minima (in mathematical terms: This refers to lowest point/value of a graph) of the 'bowl' shaped plane in 
        #   this 3D space. Even though step 3 of the (general) Gradient Descent Optimization Algorithm is much less likely 
        #   to miss the minima, it might still miss the minima if the value of the 'learning_rate_value' is set to too large, 
        #   causing the (general) Gradient Descent Optimization Algorithm to miss the minima in the first few 'steps', 
        #   even though it is designed to reduce the size of its 'steps' after each 'step' based on the gradient of the 
        #   'bowl' shaped plane in this 3D space. (We will also show an example of step 3 of the (general) Gradient Descent 
        #   Optimization Algorithm 'missing' the minima in the first few 'steps' when the value of the 'learning_rate_value' 
        #   is set to too large in the file '4. visualisation_of_the_General_Gradient_Descent_Optimization_Algorithm_missing_
        #   the_minima.py')
        print("Current iteration: {}, Current weight ('m') value: {}, Current biases ('b') value: {}, Cost Function value: {}".format(i, current_weight_m_value, current_biases_b_value, current_cost_function_value))
        


        #          Hence, the exact certain amount of every 'step' in each iteration is represented by the formulas:
        #             m' = m - (learning rate * dMSE/dm)      (for the weights, 'm')
        current_weight_m_value = current_weight_m_value - (learning_rate_value * partial_derivative_of_weight_m_with_respect_to_Cost_Function)

        #             b' = b - (learning rate * dMSE/db)      (for the biases, 'b')
        current_biases_b_value = current_biases_b_value - (learning_rate_value * partial_derivative_of_biases_b_with_respect_to_Cost_Function)
                    
        #          - m': represents the value of the weight ('m') of the line//the SVLR ML algorithm of the next iteration
        #          - m: represents the value of the weight ('m') of the line//the SVLR ML algorithm of the current iteration
        #          - b': represents the value of the biases ('b') of the line//the SVLR ML algorithm of the next iteration
        #          - b: represents the value of the biases ('b') of the line//the SVLR ML algorithm of the current iteration
        #          - (learning rate * dMSE/dm) and (learning rate * dMSE/db): represents the exact certain amount of every
        #                                                                     'step' in each iteration 
        #          - learning rate: represents a constant that controls how much the line/the SVLR ML algorithm's weights ('m')
        #                           and biases ('b') are adjusted/'step' is moved during every iteration (we will look more 
        #                           into this term and what exactly it does in code in the later files in this tutorial). 
        #                           Generally, a larger learning rate value will mean a larger adjustment/'step' in each
        #                           iteration, and likewise for a smaller learning rate value will mean a smaller 
        #                           adjustment/'step' in each iteration.



if __name__ == '__main__':
    # In the 'general_gradient_descent_optimization_algorithm()', we are directly applying mathematical operations onto 
    # the independent variables/features ('x') and dependent variables. Hence, we will be using the numpy Python library's 
    # array instead as it allow us to do mathematical operations on matrices much more easier than Python's List.
    # (e.g. You can't do 'x + some_constant_number' if 'x' is a Python's List, but you can do this if 'x' is a numpy Python
    #  library's array)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    general_gradient_descent_optimization_algorithm(x, y)


# /////////////////////////////////////////////////////////////////////////////////////////////////


# 4. Once you have reached this minima (in mathematical terms: This refers to lowest point/value of a graph) of the 'bowl' 
#    shaped plane in this 3D space. You have found the set of values of the weights ('m') and biases ('b') of the line/the 
#    SVLR ML algorithm with the minimial value of the Cost Function (MSE Cost Function in this context), since the z-axis
#    of this 3D space represents the range of values of the Cost Function (MSE Cost Function in this context)! 
   
#    Hence, you have found the weights ('m') and biases ('b') of the 'best fit line' of the set of points since the minimal 
#    value of the Cost Function (MSE Cost Function in this context) represents the 'best fit line' itself of a set of points! 


# Currently the (general) Gradient Descent Optimization Algorithm with the Mean Squared Error (MSE) Cost Function, in
# the context of the Single Variable Linear Regression (SVLR) ML algorithm, are not yet at minima. This is because the value
# of the Cost Function is still changing by a rather high amount between iterations (approximately 0.0001). Usually we will 
# only consider the (general) Gradient Descent Optimization Algorithm to have reached minima only if the value of the Cost 
# Function changes by a threshold negligible amount (e.g. 0.000000000000000000000000000000001) between iterations then it is 
# considered to have reached minima. 

# In order to get the (general) Gradient Descent Optimization Algorithm to reach the minima, there is 2 parameters/values we
# can further change, by trial and error, in order to get the set of values of the current weight ('m') and current biases ('b') 
# at the final iteration where the value of the Cost Function changes below the threshold negligible amount 
# (e.g. 0.000000000000000000000000000000001), which represents the line/the SVLR ML algorithm with the optimal/minimial value of 
# the Cost Function:
# - 'number_of_iterations'
# - 'learning_rate_value'

# And we will take the set of values of the current weight ('m') and current biases ('b') at the final iteration where the value 
# of the Cost Function changes below the threshold negligible amount (e.g. 0.000000000000000000000000000000001) between 
# iterations as the founded set of values of the weights ('m') and biases ('b') of the line/the SVLR ML algorithm with the 
# optimal/minimial value of the Cost Function.

# We will see how we can do this in the file, '3. changing_the_learning_rate_value_and_number_of_iterations_value_to_find_the_set_
# of_weights_and_biases_with_the_minimal_Cost_Function_value_in_the_General_Gradient_Descent_Optimization_Algorithm.py'

# Output:
    # Current iteration: 0, Current weight ('m') value: 0, Current biases ('b') value: 0, Cost Function value: 89.0
    # Current iteration: 1, Current weight ('m') value: 0.062, Current biases ('b') value: 0.018000000000000002, Cost Function value: 84.881304
    # Current iteration: 2, Current weight ('m') value: 0.122528, Current biases ('b') value: 0.035592000000000006, Cost Function value: 80.955185108544
    # Current iteration: 3, Current weight ('m') value: 0.181618832, Current biases ('b') value: 0.052785648000000004, Cost Function value: 77.21263768455901

    # ...

    # Current iteration: 997, Current weight ('m') value: 2.449155141761153, Current biases ('b') value: 1.3784074216500761, Cost Function value: 0.4786263787892881
    # Current iteration: 998, Current weight ('m') value: 2.449003284112507, Current biases ('b') value: 1.3789556759562092, Cost Function value: 0.4783027899709678
    # Current iteration: 999, Current weight ('m') value: 2.448851477806295, Current biases ('b') value: 1.3795037448996217, Cost Function value: 0.47797941992396514

