from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self._layers = []

    @property
    def layers(self):
        return self._layers

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass
    
