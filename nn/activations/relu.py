import numpy as np

from ..layers.base import Layer


class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        output = np.maximum(0, inputs)
        return output
    
    def backward(self, grad_output):
        dinputs = grad_output.copy()
        dinputs[self.inputs <= 0] = 0
        return dinputs
