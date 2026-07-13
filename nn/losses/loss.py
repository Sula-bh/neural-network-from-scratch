from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> None:
        pass

    def calculate(self, y_pred, y_true) -> float:
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss
