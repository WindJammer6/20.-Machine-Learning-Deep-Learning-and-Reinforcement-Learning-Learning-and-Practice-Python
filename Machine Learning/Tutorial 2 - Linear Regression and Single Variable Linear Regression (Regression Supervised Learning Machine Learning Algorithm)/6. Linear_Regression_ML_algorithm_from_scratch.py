# Implementing Linear Regression ML algorithm from scratch (both SVLR and MVLR) (copying the same style from 
# the Sci-kit learn
# library) with:
# The following functions:
# - '.fit()'
# - '.predict()'

# The following attributes:
# - '.coef_'
# - '.intercept_'

# Cost Function used: Mean Square Error (MSE) Cost Function (I wanted to implement Mean Absolute Error 
# (MAE) Cost Function as well but for some reason I just seem cant seem to get its partial derivative 
# (gradient) to work)

# Optimization Algorithm used: Batch Gradient Descent Optimization Algorithm

import numpy as np

class LinearRegression:

    def __init__(self):
        # It is convention to initialize weights to 0 (see the '.fit()' Instance method) and biases to 1 
        self.coef_ = None           # Coefficient for original features (like Sci-kit learn)
        self.intercept_ = None      # Intercept for original features (like Sci-kit learn)
        self.x_mean = None
        self.x_std = None
        self.coef_raw_ = None
        self.intercept_raw_ = None

    # MSE = -(1/n) * Σᵢ(yᵢ-ŷᵢ)²
    def mean_square_error_cost_function(self, x, y_true, y_pred, n):
        return (1/n) * np.sum(np.square(y_true - y_pred))

    # ∂MSE/∂w = -(2/n)​ * ∑​(yᵢ - ŷᵢ) * xᵢ 
    # ∂MSE/∂w = -(2/n)​ * ∑​(yᵢ - ŷᵢ) 

    # In matrix form:
    # ∂MSE/∂w = -(2/n) * Xᵀ∑(y − ŷᵢ)
    # ∂MSE/∂b = -(2/n) * ∑(y − ŷᵢ)
    def mean_square_error_cost_function_weights_grad(self, x, y_true, y_pred, n):
        return -(2/n) * np.dot(x.T, (y_true - y_pred))

    def mean_square_error_cost_function_biases_grad(self, x, y_true, y_pred, n):
        return -(2/n) * np.sum(y_true - y_pred)


    # Batch Gradient Descent Optimization algorithm
    # w = w - (learning rate, η * ∂C₀/∂w) (for the weights, w)
    # b = b - (learning rate, η * ∂C₀/∂b) (for the biases, b)
    def batch_gradient_descent_optimization_algorithm(self, x, y_true, epochs, learning_rate):
        
        # Size of the training dataset
        n = x.shape[0]

        for i in range(epochs):
            y_pred = np.dot(x, self.coef_raw_) + self.intercept_raw_

            self.coef_raw_ = self.coef_raw_ - (learning_rate * self.mean_square_error_cost_function_weights_grad(x, y_true, y_pred, n))
            self.intercept_raw_= self.intercept_raw_ - (learning_rate * self.mean_square_error_cost_function_biases_grad(x, y_true, y_pred, n))

            # Logging
            if i % 100 == 0:
                cost = self.mean_square_error_cost_function(x, y_true, y_pred, n)
                print(f"Epoch: {i} | Cost: {cost:.4f}")


    def fit(self, x, y_true, epochs, learning_rate):
        x = np.array(x)
        y_true = np.array(y_true).reshape(-1, 1)

        # Scaling of input data via Normalization to ensure the Linear Regression ML algorithms converges
        # better and more accurately
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        x_scaled = (x - self.x_mean) / self.x_std

        # Initialize weights and bias
        if self.coef_raw_ is None:
            self.coef_raw_ = np.zeros((x_scaled.shape[1], 1))  # Column vector
        if self.intercept_raw_ is None:
            self.intercept_raw_ = 1

        self.batch_gradient_descent_optimization_algorithm(x_scaled, y_true, epochs, learning_rate)

        # Transform coefficients back to original space (to match Sci-kit learn)
        self.coef_ = self.coef_raw_ / self.x_std.reshape(-1, 1)
        self.intercept_ = self.intercept_raw_ - np.sum(self.coef_raw_ * self.x_mean.reshape(-1, 1) / self.x_std.reshape(-1, 1))


    # y = m1x1 + m2x2 + ... + b
    def predict(self, x):
        x_scaled = (x - self.x_mean) / self.x_std
        y_pred = np.dot(x_scaled, self.coef_raw_) + self.intercept_raw_
        return y_pred

    
if __name__ == "__main__":
    # Using a sample dataset, 'canada_per_capita_income.csv' with 1 feature only

    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn.linear_model

    # Reading the dataset from the CSV file using the Pandas library
    dataset = pd.read_csv("canada_per_capita_income.csv")
    print(dataset)

    # Testing the output of the Linear Regression ML algorithm from scratch against the output of the 
    # Sci-kit learn library version of Linear Regression ML algorithm
    # Linear Regression ML algorithm from scratch:
    linear_regression_ML_model_scratch = LinearRegression()
    print(linear_regression_ML_model_scratch.fit(dataset[['year']], dataset['per capita income (US$)'], epochs=1000, learning_rate=0.1))
    print(f"Linear Regression ML model from scratch's weights: {linear_regression_ML_model_scratch.coef_}")
    print(f"Linear Regression ML model from scratch's bias: {linear_regression_ML_model_scratch.intercept_}")
    print(f"Prediction with the Linear Regression ML model from scratch: {linear_regression_ML_model_scratch.predict([[1700]])}")


    # Sci-kit learn library version of Linear Regression ML algorithm:
    linear_regression_ML_model = sklearn.linear_model.LinearRegression()
    print(linear_regression_ML_model.fit(dataset[['year']], dataset['per capita income (US$)']))
    print(f"Linear Regression ML model from Sci-kit learn's weights: {linear_regression_ML_model.coef_}")
    print(f"Linear Regression ML model from Sci-kit learn's bias: {linear_regression_ML_model.intercept_}")
    print(f"Prediction with the Linear Regression ML model from Sci-kit learn: {linear_regression_ML_model.predict([[1700]])}")



    # Plotting a Matploblib visual comparison of the output of the Linear Regression ML algorithm from scratch
    # and the Sci-kit learn library version of Linear Regression ML algorithm (but only works for datasets with 1 feature)
    X_plot = dataset[['year']]
    y_true = dataset['per capita income (US$)']

    # Sort X values for clean lines
    X_sorted = np.sort(X_plot, axis=0)

    # Predictions
    y_pred_custom = linear_regression_ML_model_scratch.predict(X_sorted)
    y_pred_sklearn = linear_regression_ML_model.predict(X_sorted)

    # Plot original data
    plt.scatter(X_plot, y_true, color='blue', label='Actual Data')

    # Plot Linear Regression ML model from scratch
    plt.plot(X_sorted, y_pred_custom, color='green', linewidth=2, label='Linear Regression ML model from scratch')

    # Plot Sci-kit learn library version of Linear Regression ML model
    plt.plot(X_sorted, y_pred_sklearn, color='red', linestyle='--', linewidth=2, label='Linear Regression ML model from Sci-kit learn')

    # Labels and title
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("y-axis against x-axis graph")
    plt.legend()
    plt.grid(True)
    plt.show()
