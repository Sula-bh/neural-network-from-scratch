from pprint import pprint

import numpy as np

from ..activations.relu import ReLU
from ..activations.softmax import Softmax
from ..initializers.random_normal import RandomNormalInitializer
from ..layers.dense import Dense
from ..losses.categorical_cross_entropy import CategoricalCrossEntropy
from ..models.sequential import Sequential
from ..utils.gradient_check import gradient_check


def main():
    rng = np.random.default_rng(seed=42)
    X_test = rng.random(size=(3, 4))
    y_test = rng.integers(low=0, high=3, size=(3,))

    model = Sequential(
        [
            Dense(4, 3, RandomNormalInitializer(seed=42)),
            ReLU(),
            Dense(3, 3, RandomNormalInitializer(seed=1)),
            Softmax(),
        ]
    )

    loss = CategoricalCrossEntropy()

    print("Running Gradient Check")

    report = gradient_check(model, loss, X_test, y_test)

    pprint(report)


if __name__ == "__main__":
    main()
