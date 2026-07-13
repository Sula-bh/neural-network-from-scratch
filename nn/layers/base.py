from abc import ABC, abstractmethod


class Layer(ABC):
    @property
    def parameters(self):
        return []
    
    @property
    def gradients(self):
        return []

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass
