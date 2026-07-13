from .base import Model


class Sequential(Model):
    def __init__(self, layers=None):
        super().__init__()
        self._layers = layers if layers is not None else []

    def add(self, layer):
        self._layers.append(layer)

    def forward(self, inputs):
        current_val = inputs
        for layer in self.layers:
            current_val = layer.forward(current_val)
        return current_val
        
    def backward(self, grad_output):
        current_grad = grad_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        return current_grad