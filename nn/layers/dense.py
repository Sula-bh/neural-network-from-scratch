import numpy as np

from ..initializers.random_normal import RandomNormalInitializer
from .base import Layer


class Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, initializer=None):
        if initializer is None:
            initializer = RandomNormalInitializer()
        self.weights = initializer.initialize(shape=(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.dweights = None
        self.dbiases = None

    @property
    def parameters(self):
        return [self.weights, self.biases]
    
    @property
    def gradients(self):
        return [self.dweights, self.dbiases]

    def forward(self, inputs):
        self.inputs = inputs
        output = inputs @ self.weights + self.biases
        return output

    def backward(self, grad_output):
        if self.inputs is None:
            raise ValueError("Cannot perform backward pass without stored inputs")
        self.dweights = self.inputs.T @ grad_output
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        dinputs = grad_output @ self.weights.T
        return dinputs