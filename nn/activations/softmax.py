import numpy as np

from ..layers.base import Layer


class Softmax(Layer):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, grad_output):
        dinputs = np.empty_like(grad_output)

        for index, (single_output, single_grad_output) in enumerate(
            zip(self.output, grad_output)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            dinputs[index] = np.dot(jacobian_matrix, single_grad_output)
        return dinputs
