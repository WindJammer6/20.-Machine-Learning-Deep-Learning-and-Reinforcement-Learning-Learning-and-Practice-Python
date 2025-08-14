# Implementing Binary Logistic Regression ML algorithm from scratch (only BLR, does not handle MLR) (copying 
# the same style from the Sci-kit learn
# library) with:
# The following functions:
# - '.fit()'
# - '.predict()'

# The following attributes:
# - '.coef_'
# - '.intercept_'

# Cost Function used: Binary Logistic Loss/Binary Cross-Entropy Cost Function

# Optimization Algorithm used: Batch Gradient Descent Optimization Algorithm

# (Unfortunately I tried to integrate Multinomial Logistic Regression ML algorithm under the same 
# 'LogisticRegression' class but I found it too tedious to do that (had trouble implementing softmax and 
# handling both binary class and multiclass scenarios in the same 'LogisticRegression' class))

import numpy as np

class LogisticRegression:
    def __init__(self):
        # It is convention to initialize weights to 0 (see the '.fit()' Instance method) and biases to 1 
        self.coef_ = None           # Coefficient for original features (like Sci-kit learn)
        self.intercept_ = None      # Intercept for original features (like Sci-kit learn)
        self.x_mean = None
        self.x_std = None
        self.coef_raw_ = None
        self.intercept_raw_ = None

    # BinaryCrossEntropy = -(1/n) * Σ[yᵢ*log(ŷᵢ) + (1-yᵢ) * log(1-ŷᵢ)]
    def binary_logistic_loss_cost_function(self, x, y_true, y_pred, n):
        epsilon = 1e-15
        y_pred_new = [max(i, epsilon) for i in y_pred]
        y_pred_new = [min(i, 1-epsilon) for i in y_pred_new]
        y_pred_new = np.array(y_pred_new)
        return -np.mean(y_true * np.log(y_pred_new) + (1-y_true) * np.log(1-y_pred_new))    # According to the mathematical 
                                                                                            # formula of the Binary Logistic Loss or Binary 
                                                                                            # Cross Entropy Cost Function

    # After using backpropogation chain rule formula:
    # ∂C₀/∂w = (1/n)​ * ∑​(y − ŷᵢ) * x​ᵢ
    # ∂C₀/∂b = (1/n)​ * ∑​(y − ŷᵢ)

    # In matrix form:
    # ∂C₀/∂w = (1/n)​ * Xᵀ∑​(y − ŷᵢ) * x​ᵢ
    # ∂C₀/∂b = (1/n)​ * ∑​(y − ŷᵢ)
    def binary_logistic_loss_cost_function_weights_grad(self, x, y_true, y_pred, n):
        return -(1/n) * np.dot(x.T, (y_true - y_pred))

    def binary_logistic_loss_cost_function_biases_grad(self, x, y_true, y_pred, n):
        return -np.mean(y_true - y_pred)


    # Batch Gradient Descent Optimization algorithm
    # w = w - (learning rate, η * ∂C₀/∂w) (for the weights, w)
    # b = b - (learning rate, η * ∂C₀/∂b) (for the biases, b)
    def batch_gradient_descent_optimization_algorithm(self, x, y_true, epochs, learning_rate):
        
        # Size of the training dataset
        n = x.shape[0]

        for i in range(epochs):
            weighted_sum = np.dot(x, self.coef_raw_) + self.intercept_raw_
            y_pred = 1 / (1 + np.exp(-weighted_sum))

            self.coef_raw_ = self.coef_raw_ - (learning_rate * self.binary_logistic_loss_cost_function_weights_grad(x, y_true, y_pred, n))
            self.intercept_raw_= self.intercept_raw_ - (learning_rate * self.binary_logistic_loss_cost_function_biases_grad(x, y_true, y_pred, n))

            # Logging
            if i % 100 == 0:
                cost = self.binary_logistic_loss_cost_function(x, y_true, y_pred.flatten(), n)
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



    # sigmoid(z) = 1 / (1 + 𝑒^−z)
    # where,
    # - '𝑧' = m * age + b
    def predict(self, x, threshold=0.5):
        x_scaled = (x - self.x_mean) / self.x_std

        weighted_sum = np.dot(x_scaled, self.coef_raw_) + self.intercept_raw_
        y_pred = 1 / (1 + np.exp(-weighted_sum))
        y_pred_class = (y_pred >= threshold).astype(int)

        return y_pred_class


if __name__ == "__main__":
    # Using a sample dataset, 'bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_dependent_variable.csv' with 1 feature only

    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn.linear_model

    # Reading the dataset from the CSV file using the Pandas library
    dataset = pd.read_csv("bought_insurance_or_not_bought_insurance_with_single_independent_variable_and_binary_categorical_dependent_variable.csv")
    print(dataset)

    # Testing the output of the Logistic Regression ML algorithm from scratch against the output of the 
    # Sci-kit learn library version of Logistic Regression ML algorithm
    # Logistic Regression ML algorithm from scratch:
    logistic_regression_ML_model_scratch = LogisticRegression()
    print(logistic_regression_ML_model_scratch.fit(dataset[['age']], dataset['bought_insurance'], epochs=1000, learning_rate=0.1))
    print(f"Logistic Regression ML model from scratch's weights: {logistic_regression_ML_model_scratch.coef_}")
    print(f"Logistic Regression ML model from scratch's bias: {logistic_regression_ML_model_scratch.intercept_}")
    print(f"Prediction with the Logistic Regression ML model from scratch: {logistic_regression_ML_model_scratch.predict([[50]])}")


    # Sci-kit learn library version of Logistic Regression ML algorithm:
    logistic_regression_ML_model = sklearn.linear_model.LogisticRegression()
    print(logistic_regression_ML_model.fit(dataset[['age']], dataset['bought_insurance']))
    print(f"Logistic Regression ML model from Sci-kit learn's weights: {logistic_regression_ML_model.coef_}")
    print(f"Logistic Regression ML model from Sci-kit learn's bias: {logistic_regression_ML_model.intercept_}")
    print(f"Prediction with the Logistic Regression ML model from Sci-kit learn: {logistic_regression_ML_model.predict([[50]])}")



    # Plotting a Matploblib visual comparison of the output of the Logistic Regression ML algorithm from scratch
    # and the Sci-kit learn library version of Logistic Regression ML algorithm (but only works for datasets with 1 feature)
    X_plot =  dataset[['age']]
    y_true = dataset['bought_insurance']

    # Sort X values for clean lines
    X_sorted = np.sort(X_plot, axis=0)

    # Predictions
    y_pred_custom = logistic_regression_ML_model_scratch.predict(X_sorted)
    y_pred_sklearn = logistic_regression_ML_model.predict(X_sorted)

    # Plot original data
    plt.scatter(X_plot, y_true, color='blue', label='Actual Data')

    # Plot Logistic Regression ML model from scratch
    plt.plot(X_sorted, y_pred_custom, color='green', linewidth=2, label='Logistic Regression ML model from scratch')

    # Plot Sci-kit learn library version of Logistic Regression ML model
    plt.plot(X_sorted, y_pred_sklearn, color='red', linestyle='--', linewidth=2, label='Logistic Regression ML model from Sci-kit learn')

    # Labels and title
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("y-axis against x-axis graph")
    plt.legend()
    plt.grid(True)
    plt.show()
