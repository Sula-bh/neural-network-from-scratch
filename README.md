# Neural Network from Scratch

A small, educational neural network library built with pure Python and NumPy. This project demonstrates how a neural network can be implemented from the ground up, including dense layers, activation functions, loss functions, an optimizer, and a simple sequential model API.

## Why this project?

This repository is designed for learning and experimentation. Instead of relying on a high-level framework, it breaks the training pipeline into small components so you can inspect how:

- forward propagation works
- backpropagation is computed
- gradients are updated
- layers and models are composed

## Features

- Dense layers with learnable weights and biases
- ReLU and Softmax activations
- Categorical cross-entropy and mean squared error losses
- Accuracy metric
- Stochastic Gradient Descent optimizer
- Gradient checking utilities for debugging
- An XOR experiment to see the network train end-to-end

## Installation

From the project root, install the package in editable mode:

```bash
pip install -e .
```

## Run the XOR example

A ready-made experiment is included for the XOR problem:

```bash
python experiments/xor/xor.py
```

## Project Structure

```text
nn/
  activations/
  initializers/
  layers/
  losses/
  metrics/
  models/
  optimizers/
  tests/
experiments/
  xor/
```

