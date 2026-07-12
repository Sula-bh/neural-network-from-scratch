import numpy as np

from ..initializers.random_normal import RandomNormalInitializer
from .base import Layer


class Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, initializer=None):
        if initializer is None:
            initializer = RandomNormalInitializer()
        self.weights = initializer.initialize(shape=(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        output = inputs @ self.weights + self.biases
        return output

    def backward(self, grad_output):
        self.dweights = self.inputs.T @ grad_output
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        dinputs = grad_output @ self.weights.T
        return dinputs