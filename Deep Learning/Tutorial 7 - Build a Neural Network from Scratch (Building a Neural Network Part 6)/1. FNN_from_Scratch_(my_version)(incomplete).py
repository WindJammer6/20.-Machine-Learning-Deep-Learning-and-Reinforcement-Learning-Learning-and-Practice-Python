# Implementing Feedforward Neural Network (FNN) from scratch (copying the same style from the TensorFlow
# library) with:
# The following functions:
# - '.compile()'
# - '.fit()'
# - '.predict()'
# - 'get_weights()'

# Cost Function used: Binary Logistic Loss/Binary Cross-Entropy Cost Function

# Optimization Algorithm used: 
# - Batch Gradient Descent Optimization Algorithm
# - Stochastic Gradient Descent Optimization Algorithm
# - Mini-Batch Gradient Descent Optimization Algorithm

# Reference source(s):
# - https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/ (freeCodeCamp)
# - https://github.com/lionelmessi6410/Neural-Networks-from-Scratch (Chia-Hung Yuan on GitHub)


# Disclaimer:
# Unfortunately implementing a Neural Network from scratch turned out to be wayyyyyy more complicated than I thought
# this was what I did so far... but couldnt put in the time to finish the full logic... hence I sought help from
# ChatGPT based on the code I have here to complete it for me


import numpy as np
import random

class Dense:
    def __init__(self, number_of_neurons, input_shape, activation):
        self.number_of_neurons = number_of_neurons
        self.input_shape = input_shape
        self.activation = activation

class FNN:
    def __init__(self):
        self.optimizer = None
        self.loss = None
        self.layers = None
        self.caches = None      # To store the weights and biases for each layer in a sort of 'memory' during
                                # forward propogation to be used later during back propogation

        # It is convention to initialize weights randomly (see the '.Sequential()' 
        # Instance method) and biases to 1
        self.weights_and_biases = None
    
    ##############################
    # -- Activation Functions -- #
    ##############################
    # a = σ(z) = sigmoid(z) = 1 / (1 + 𝑒^−z)
    def sigmoid_activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    # ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = σ′(z) = a * (1 − a)         (a = σ(z) = tanh(z))
    def sigmoid_activation_function_grad(self, a):
        return a * (1 - a)
    
    # a = σ(z) = tanh(z) = ( 2 / (1 + e^(−2z)) ) - 1  OR  tanh(z) = 2 * sigmoid(2z) − 1  
    def tanh_activation_function(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    # ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = σ'(z) = 1 − a^2             (a = σ(z) = tanh(z))
    def tanh_activation_function_grad(self, a):
        return (1 - pow(a, 2))

    # a = σ(z) = ReLU(z) = max(0, z)
    def relu_activation_function(self, z):
        return np.maximum(0, z)
    
    # ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = σ′(z) = {1  , if z > 0      (a = σ(z) = ReLU(z))
    #                        0  , otherwise
    def relu_activation_function_grad(self, a):
        return (a > 0).astype(float)
    
    # a = σ(z) = LeakyReLU(z) = max(0.01z, z)
    def leaky_relu_activation_function(self, z):
        return np.maximum(0.01 * z, z)
    
    # ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = σ′(z) = {1     , if z > 0    (a = σ(z) = LeakyReLU(z))
    #                        0.01  , otherwise
    def leaky_relu_activation_function_grad(self, z):
        return np.where(z > 0, 1.0, 0.01)

    ########################
    # -- Cost Functions -- #
    ########################
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


    #################################
    # -- Optimization Algorithms -- #
    #################################
    def batch_gradient_descent_optimization_algorithm(self, x, y_true, epochs, learning_rate=0.01):
        number_of_features = x.shape[1]

        # Initializing the weights and biases
        weights = np.ones(shape=(number_of_features))
        biases = 0
        total_samples = x.shape[0]

        cost_list = []
        epoch_list = []

        cost = self.choose_cost_function(y_pred, y)
        cost_history.append(cost)

        for i in range(epochs):
            y_predicted = np.dot(weights, scaled_x.T) + biases     # w1 * area + w2 * bedrooms + biases

            weights_grad = -(2 / total_samples) * (x.T.dot(y_true - y_predicted))
            biases_grad = -(2 / total_samples) * np.sum(y_true - y_predicted)

            weights = weights - learning_rate * weights_grad
            biases = biases - learning_rate * biases_grad

            cost = np.mean(np.square(y_true - y_predicted))

            if i % 10 == 0:
                cost_list.append(cost)
                epoch_list.append(i)

        return weights, biases, cost, cost_list, epoch_list

    def stochastic_gradient_descent_optimization_algorithm(self, x, y_true, epochs, learning_rate=0.01):
        number_of_features = x.shape[1]

        # Initializing the weights and biases
        weights = np.ones(shape=(number_of_features))
        biases = 0
        total_samples = x.shape[0]

        cost_list = []
        epoch_list = []

        for i in range(epochs):
            random_index = random.randint(0, total_samples-1)
            sample_x = x[random_index]
            sample_y = y_true[random_index]

            y_predicted = np.dot(weights, sample_x.T) + biases     # w1 * area + w2 * bedrooms + biases

            weights_grad = -(2 / 1) * (sample_x.T.dot(sample_y - y_predicted))   # denominator is not 'total_samples' but 1 instead since Stochastic
            biases_grad = -(2 / 1) * (sample_y - y_predicted)                    # Gradient Descent only takes 1 training example per epoch

            weights = weights - learning_rate * weights_grad
            biases = biases - learning_rate * biases_grad

            cost = np.square(sample_y - y_predicted)        # we dont do np.mean here for Stochastic Gradient Descent

            if i % 100 == 0:
                cost_list.append(cost)
                epoch_list.append(i)

        return weights, biases, cost, cost_list, epoch_list

    def mini_batch_gradient_descent_optimization_algorithm(self, x, y_true, epochs, learning_rate=0.01):
        number_of_features = x.shape[1]

        # Initializing the weights and biases
        weights = np.ones(shape=(number_of_features))
        biases = 0
        total_samples = x.shape[0]

        cost_list = []
        epoch_list = []

        batch_size = 5
        num_batches = int(total_samples/batch_size)

        for i in range(epochs):
            random_indices = np.random.permutation(total_samples)
            X_tmp = x[random_indices]
            y_tmp = y_true[random_indices]
            
            for j in range(0,total_samples,batch_size):
                Xj = X_tmp[j:j+batch_size]
                yj = y_tmp[j:j+batch_size]
                y_predicted = np.dot(weights, Xj.T) + biases
                
                weights_grad = -(2/len(Xj))*(Xj.T.dot(yj-y_predicted))
                biases_grad = -(2/len(Xj))*np.sum(yj-y_predicted)
                
                weights = weights - learning_rate * weights_grad
                biases = biases - learning_rate * biases_grad
                    
                cost = np.mean(np.square(yj-y_predicted)) # MSE (Mean Squared Error)

            if i % 50 == 0:        # we do modulo 50 here instead of 100 cuz theres less epoch in Mini-Batch Gradient Descent compared to Stochastic Gradient Descent
                cost_list.append(cost)
                epoch_list.append(i)

        return weights, biases, cost, cost_list, epoch_list


    ###################################
    # -- Shown/Usable Main methods -- #
    ###################################
    # Mimicing TensorFlow implementation, this '.Sequential()' Instance method takes in a list of 'Dense'
    # objects
    def Sequential(self, layers):
        self.layers = layers
        self.caches = []
        self.weights_and_biases = {}

        for l in range(len(layers)):
            # Xavier initialization
            self.weights_and_biases['W'+str(l)] = np.random.randn(self.l.number_of_neurons, self.l.input_shape[0]) * np.sqrt(1. / self.l.input_shape[0])
            self.weights_and_biases['b'+str(l)] = np.ones((self.l.number_of_neurons, 1))

    # Omitting trying to implment TensorFlow's '.compile()' function's 'metrics' parameter feature... 
    # (seems tricky to mimic)
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, x, y, epochs, learning_rate=0.01):
        cost_history = []

        for i in range(epochs):
            y_pred = self.forward_propogation(x)

            grads = self.back_propogation(y_pred, y)

            self.batch_gradient_descent_optimization_algorithm(self.weights_and_biases, grads, learning_rate)

    def predict(self, x):
        # Using trained weights and biases
        return self.forward_propogation(x)

    def get_weights(self):
        return self.weights_and_biases

    
    #############################
    # -- Hidden Main methods -- #
    #############################
    def choose_activation_function(self, z):
        if self.activation == 'sigmoid':
            return self.sigmoid_activation_function(z)
        elif self.activation == 'tanh':
            return self.tanh_activation_function(z)
        elif self.activation == 'relu':
            return self.relu_activation_function(z)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu_activation_function(z)
        else:
            return z  # Linear Activation Function

    def choose_optimization_algorithm(self, x, y_true, epochs, learning_rate):
        if self.optimizer == 'sgd':
            return self.stochastic_gradient_descent_optimization_algorithm(x, y_true, epochs, learning_rate)
        elif self.optimizer == 'mbgd':
            return self.mini_batch_gradient_descent_optimization_algorithm(x, y_true, epochs, learning_rate)
        else:
            return self.batch_gradient_descent_optimization_algorithm()  # Batch Optimization Algorithm

    def choose_cost_function(self, x, y_true, y_pred, n):
        if self.loss == 'binary_crossentropy':
            return self.binary_logistic_loss_cost_function(x, y_true, y_pred, n)
        else:
            return self.binary_logistic_loss_cost_function()  # Binary Logistic Loss/Binary Cross-Entropy Cost Function

    # - 'x' stands for the inputs
    def forward_propogation(self, x):
        A = x                                   # input to first layer i.e. training data
        L = len(self.weights_and_biases) // 2   # number of layers (since each layer has W and b)

        for l in range(1, L+1):
            A_prev = A

            # Linear Hypothesis
            Z = np.dot(self.weights_and_biases['W'+str(l)], A_prev) + self.weights_and_biases['b'+str(l)] 

            # Storing the linear cache
            linear_cache = (A_prev, self.weights_and_biases['W'+str(l)], self.weights_and_biases['b'+str(l)]) 

            # Applying sigmoid on linear hypothesis
            A, activation_cache = self.sigmoid_activation_function(Z) 

            # storing the both linear and activation cache
            cache = (linear_cache, activation_cache)
            self.caches.append(cache)

        return A  

    
    def one_layer_backward(self, dA):
        linear_cache, activation_cache = self.caches

        Z = activation_cache
        dZ = dA * self.sigmoid_activation_function(Z)*(1 - self.sigmoid_activation_function(Z)) # The derivative of the sigmoid function

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    # - 'A_Last' stands for the output layer predictions
    # - 'm stands for the number of training examples
    # - 'Y' stands for the ground truth labels
    def back_propogation(self, A_Last, Y):
        gradients = {}
        L = len(self.caches)
        m = A_Last.shape[1]
        # Reshape Y to ensure it's the same shape as A_Last for element-wise operations
        Y = Y.reshape(A_Last.shape)
    
        dA_Last = self.binary_logistic_loss_cost_function_weights_grad(x, y_true, y_pred, n)

        current_cache = self.caches[L-1]
        gradients['dA'+str(L-1)], gradients['dW'+str(L-1)], gradients['db'+str(L-1)] = self.one_layer_backward(dA_Last, current_cache)

        for l in reversed(range(L-1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = self.one_layer_backward(gradients["dA" + str(l+1)], current_cache)
            gradients["dA" + str(l)] = dA_prev_temp
            gradients["dW" + str(l + 1)] = dW_temp
            gradients["db" + str(l + 1)] = db_temp

        return gradients


if __name__ == "__main__":
    import tensorflow as tf
    import pandas as pd
    from tensorflow import keras
    from sklearn.model_selection import train_test_split

    ######################
    # Data Preprocessing #
    ######################
    df = pd.read_csv("insurance_data_synthetic.csv")
    print(df.head())
    # print(df.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(df[['age', 'affordibility']], df.bought_insurance, test_size=0.2, random_state=25)
    # print(x_train)
    # print(len(x_train))

    # Scaling by normalization:
    x_train_scaled = x_train.copy()
    x_train_scaled['age'] = (x_train_scaled['age'] - 0) / (100 - 0)

    x_test_scaled = x_test.copy()
    x_test_scaled['age'] = x_test_scaled['age'] / 100

    # print(x_train_scaled)
    # print(x_test_scaled)

    ################################
    # Training the custom made FNN #
    ################################
    simple_feedforward_neural_network_DL_model_scratch = FNN()
    simple_feedforward_neural_network_DL_model_scratch.Sequential([
        Dense(2, input_shape=(2, ), activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Omitting trying to implment TensorFlow's '.compile()' function's 'metrics' parameter feature... 
    # (seems tricky to mimic)
    simple_feedforward_neural_network_DL_model_scratch.compile(
        optimizer='sgd',
        loss='binary_crossentropy'
    )

    simple_feedforward_neural_network_DL_model_scratch.fit(x_train_scaled, y_train, epochs=5)

    y_predicted = simple_feedforward_neural_network_DL_model_scratch.predict(x_test_scaled)
    print(y_predicted)
    y_predicted_classes = (y_predicted > 0.5).astype(int)
    print(y_predicted_classes.flatten())  # Flatten for a nicer 1D display

    weights_and_biases_scratch = simple_feedforward_neural_network_DL_model_scratch.get_weights()     # weights and biases
    for i, (w, b) in enumerate(weights_and_biases_scratch):
        print(f"Layer {i+1}:")
        print(f"  Weights shape: {w.shape}")
        print(f"  Weights:\n{w}")
        print(f"  Biases shape: {b.shape}")
        print(f"  Biases:\n{b}\n")


    ##################################
    # Training a FNN with TensorFlow #
    ##################################
    simple_feedforward_neural_network_DL_model = keras.Sequential([
        keras.layers.Dense(2, input_shape=(2, ), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Here, we will simply just try with the 'adam' optimizer. 'sparse_categorical_crossentropy' loss function and
    # 'accuracy' metrics. It is up to you to play around with the different types of hyper-parameters, to see which oens
    # give you a model of the highest accuracy.
    simple_feedforward_neural_network_DL_model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    simple_feedforward_neural_network_DL_model.fit(x_train_scaled, y_train, epochs=5)

    y_predicted = simple_feedforward_neural_network_DL_model.predict(x_test_scaled)
    print(y_predicted)
    y_predicted_classes = (y_predicted > 0.5).astype(int)
    print(y_predicted_classes.flatten())  # Flatten for a nicer 1D display

    weights_and_biases = simple_feedforward_neural_network_DL_model.get_weights()     # weights and biases
    num_layers = len(weights_and_biases) // 2
    for i in range(num_layers):
        w = weights_and_biases[2 * i]
        b = weights_and_biases[2 * i + 1]
        print(f"Layer {i+1}:")
        print(f"  Weights shape: {w.shape}")
        print(f"  Weights:\n{w}")
        print(f"  Biases shape: {b.shape}")
        print(f"  Biases:\n{b}\n")