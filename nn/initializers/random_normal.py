import numpy as np

from .base import Initializer


class RandomNormalInitializer(Initializer):
    def __init__(self, mean: float = 0, std: float = 0.01, seed: int | None = None):
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng(seed)

    def initialize(self, shape: tuple[int, ...]):
        return self.rng.normal(self.mean, self.std, size=shape)
