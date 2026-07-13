import numpy as np

from .base import Metric


class Accuracy(Metric):
    def compute(self, predictions, targets):
        if len(predictions.shape) == 2:
            predictions = np.argmax(predictions, axis=1)
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)
        accuracy = np.mean(predictions == targets)
        return accuracy
