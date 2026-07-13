from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, predictions, targets) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> None:
        pass

    def calculate(self, predictions, targets) -> float:
        sample_losses = self.forward(predictions, targets)
        data_loss = np.mean(sample_losses)
        return data_loss
