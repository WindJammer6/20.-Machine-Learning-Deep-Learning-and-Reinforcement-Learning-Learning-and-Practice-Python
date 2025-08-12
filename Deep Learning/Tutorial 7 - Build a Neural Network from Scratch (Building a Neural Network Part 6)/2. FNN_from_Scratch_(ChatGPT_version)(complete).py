# Implementing Feedforward Neural Network (FNN) from scratch (copying the same style from the TensorFlow
# library) with:
# The following functions:
# - '.compile()'
# - '.fit()'
# - '.predict()'
# - 'get_weights()'

# Cost Function used: Binary Logistic Loss/Binary Cross-Entropy Cost Function

# Optimization Algorithm used: Batch Gradient Descent Optimization Algorithm

# Reference source(s):
# - https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/ (freeCodeCamp)
# - https://github.com/lionelmessi6410/Neural-Networks-from-Scratch (Chia-Hung Yuan on GitHub)


# Disclaimer:
# Unfortunately even this ChatGPT's implementation a Neural Network from scratch dosent work very well either. It only
# could run for binary predictions and not for multiclass predictions (e.g. the digits image dataset from Sci-kit learn)
# since when I tried to use softmax the gradients were exploding for some reason... 
# 
# Even for binary predictions it dosent work very well and is not very accurate at all. Much less accurate compared to
# TensorFlow's implementation 


import numpy as np

# Dense Layer
class Dense:
    def __init__(self, number_of_neurons, input_shape=None, activation='linear'):
        self.number_of_neurons = number_of_neurons
        self.input_shape = input_shape
        self.activation = activation
        self.weights = None
        self.biases = None
        self.z = None
        self.a_prev = None

# Feedforward Neural Network
class FNN:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer_fn = None

    def Sequential(self, layers):
        for i, layer in enumerate(layers):
            if layer.input_shape is None:
                if i == 0:
                    raise ValueError("The first layer must have input_shape defined.")
                prev_layer = layers[i - 1]
                layer.input_shape = (prev_layer.number_of_neurons,)

            # Initialize weights after input_shape is available
            in_dim = layer.input_shape[0]
            out_dim = layer.number_of_neurons
            layer.weights = np.random.randn(out_dim, in_dim) * np.sqrt(1. / in_dim)
            layer.biases = np.ones((out_dim, 1))

        self.layers = layers

    def compile(self, optimizer, loss):
        self.loss = loss
        if optimizer == 'sgd':
            self.optimizer_fn = self._sgd
        elif optimizer == 'bgd':
            self.optimizer_fn = self._bgd
        else:
            raise ValueError("Unsupported optimizer")

    ##############################
    # -- Activation Functions -- #
    ##############################
    def choose_activation_function(self, z, func):
        if func == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif func == 'tanh':
            return np.tanh(z)
        elif func == 'relu':
            return np.maximum(0, z)
        elif func == 'leaky_relu':
            return np.maximum(0.01 * z, z)
        elif func == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            return z

    def chosen_activation_function_grad(self, a, func):
        if func == 'sigmoid':
            return a * (1 - a)
        elif func == 'tanh':
            return 1 - a**2
        elif func == 'relu':
            return (a > 0).astype(float)
        elif func == 'leaky_relu':
            return np.where(a > 0, 1.0, 0.01)
        else:
            return np.ones_like(a)

    ########################
    # -- Cost Functions -- #
    ########################
    def binary_crossentropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_crossentropy_grad(self, y_true, y_pred):
        return y_pred - y_true
    
    #############################
    # -- Hidden Main methods -- #
    #############################
    def forward_propogation(self, x):
        A = x.T
        for layer in self.layers:
            Z = np.dot(layer.weights, A) + layer.biases
            A_prev = A
            A = self.choose_activation_function(Z, layer.activation)
            layer.z = Z
            layer.a_prev = A_prev
        return A

    def back_propogation(self, y_pred, y_true):
        grads = []
        m = y_true.shape[0]
        y_true = y_true.T
        dA = self.binary_crossentropy_grad(y_true, y_pred)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            Z = layer.z
            A_prev = layer.a_prev
            A = self.choose_activation_function(Z, layer.activation)
            activation_derivative = self.chosen_activation_function_grad(A, layer.activation)

            dZ = dA * activation_derivative
            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(layer.weights.T, dZ)

            grads.insert(0, (dW, db))

        return grads

    def update_weights(self, grads, learning_rate):
        for i, layer in enumerate(self.layers):
            dW, db = grads[i]
            layer.weights -= learning_rate * dW
            layer.biases -= learning_rate * db

    #################################
    # -- Optimization Algorithms -- #
    #################################
    def _bgd(self, x, y, epochs, learning_rate):
        y = y.to_numpy().reshape(-1, 1)
        for epoch in range(epochs):
            y_pred = self.forward_propogation(x)
            loss = self.binary_crossentropy(y.T, y_pred)
            grads = self.back_propogation(y_pred, y)
            self.update_weights(grads, learning_rate)
            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss}")

    def _sgd(self, x, y, epochs, learning_rate):
        m = x.shape[0]
        for epoch in range(epochs):
            for i in range(m):
                xi = x[i:i+1]
                yi = y[i:i+1].to_numpy().reshape(-1, 1)
                y_pred = self.forward_propogation(xi)
                loss = self.binary_crossentropy(yi.T, y_pred)
                grads = self.back_propogation(y_pred, yi)
                self.update_weights(grads, learning_rate)
            print(f"Epoch {epoch+1}: Loss = {loss}")

    ###################################
    # -- Shown/Usable Main methods -- #
    ###################################
    def fit(self, x, y, epochs, learning_rate=0.01):
        self.optimizer_fn(x, y, epochs, learning_rate)

    def predict(self, x):
        return self.forward_propogation(x).T  # Transpose back to (n_samples, 1)

    def get_weights(self):
        return [(layer.weights, layer.biases) for layer in self.layers]


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