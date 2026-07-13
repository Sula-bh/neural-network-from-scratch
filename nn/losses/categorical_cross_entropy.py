import numpy as np

from .loss import Loss


class CategoricalCrossEntropy(Loss):
    def forward(self, predictions, targets):
        self.targets = targets
        batch_size = predictions.shape[0]
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        self.predictions = predictions_clipped

        if len(targets.shape) == 1:
            correct_confidences = predictions_clipped[range(batch_size), targets]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self):
        batch_size = self.predictions.shape[0]
        n_features = self.predictions.shape[1]
        if len(self.targets.shape) == 1:
            self.targets = np.eye(n_features)[self.targets.astype(np.int64)]

        dinputs = -self.targets / self.predictions
        return dinputs / batch_size
