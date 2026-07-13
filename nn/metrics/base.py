from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def compute(self, predictions, targets):
        pass
