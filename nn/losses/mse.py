import numpy as np

from .loss import Loss


class MSE(Loss):
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        mse = 0.5 * np.mean(np.square(predictions - targets), axis=1)
        return mse
    
    def backward(self):
        batch_size = self.predictions.shape[0]
        n_features = self.predictions.shape[1]
        return (self.predictions - self.targets) / (n_features * batch_size)