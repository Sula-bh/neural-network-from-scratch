import numpy as np

from ..activations.softmax import Softmax
from .categorical_cross_entropy import CategoricalCrossEntropy
from .loss import Loss


class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        self._activation = Softmax()
        self._loss = CategoricalCrossEntropy()

    def forward(self, logits, targets):
        self.targets = targets
        self.predictions = self._activation.forward(logits)
        return self._loss.forward(self.predictions, targets)

    def backward(self):
        batch_size = self.predictions.shape[0]
        if len(self.targets.shape) == 2:
            self.targets = np.argmax(self.targets, axis=1)

        dinputs = self.predictions.copy()
        dinputs[range(batch_size), self.targets] -= 1
        return dinputs / batch_size
