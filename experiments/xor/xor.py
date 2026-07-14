import numpy as np

from nn.activations.relu import ReLU
from nn.activations.softmax import Softmax
from nn.initializers.random_normal import RandomNormalInitializer
from nn.layers.dense import Dense
from nn.losses.categorical_cross_entropy import CategoricalCrossEntropy
from nn.metrics.accuracy import Accuracy
from nn.models.sequential import Sequential
from nn.optimizers.SGD import SGD

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Sequential(
    [
        Dense(2, 4, RandomNormalInitializer(seed=1)),
        ReLU(),
        Dense(4, 2, RandomNormalInitializer(seed=42)),
        Softmax(),
    ]
)

loss_fn = CategoricalCrossEntropy()

optimizer = SGD(learning_rate=0.1)

accuracy_fn = Accuracy()

for epoch in range(1001):
    predictions = model.forward(X)
    loss = loss_fn.calculate(predictions, y)
    accuracy = accuracy_fn.compute(predictions, y)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")

    model.backward(loss_fn.backward())

    optimizer.step(model.layers)

current = X

for layer in model.layers:
    current = layer.forward(current)
    print(layer.__class__.__name__)
    print(current)

for i, layer in enumerate(model.layers):
    if layer.parameters:
        print(f"\nLayer {i}")
        print("Weights:")
        print(layer.weights)
        print("Biases:")
        print(layer.biases)


print("\nThe predicted probabilities are:\n", predictions)
print("\nThe predictions are:\n", np.argmax(predictions, axis=1))