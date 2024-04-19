import numpy as np


class Dense:

    def __init__(self, inputs, size) -> None:
        self.weights = np.random.random((size, inputs))
        self.bias = np.random.random((size, 1))

    def __call__(self, X):
        return np.dot(self.weights, X) + self.bias


class NeuralNetwork:

    def __init__(self, input_size, output_size):
        self.layer1 = Dense(input_size, output_size)

    def __call__(self, X):
        return self.layer1(X)
