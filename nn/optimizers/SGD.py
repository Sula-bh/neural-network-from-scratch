from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=1.0):
        super().__init__(learning_rate)

    def step(self, layers):
        for layer in layers:
            for parameter, gradient in zip(layer.parameters, layer.gradients):
                parameter -= self.learning_rate * gradient