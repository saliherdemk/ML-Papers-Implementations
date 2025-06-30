"""
Dense1 -> Dense2 -> MSE

For Dense1
----
o1 = w1 * x1 + b1
o2 = w2 * (w1 * x1 + b1) + b2
L = 1/2 (o2 - y) ^ 2

∂L / ∂w1 = ∂L / ∂o2 * ∂o2 / ∂o1 * ∂o1 / ∂w1
∂L / ∂o2 = o2 - y
∂o2 / ∂o1 = w2
∂o1 / ∂w1 = x1

∂L / ∂w1 = (o2 - y) * w2 * x1
            |          |
        grad_output for dense 1
∂L / ∂w2 = (o2 - y) * x2
           |       |
        grad_output for dense 2
"""

import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.random.randn(output_size) * 0.1

    def forward(self, x):
        self.input = x
        return self.weights @ x + self.biases

    def backward(self, grad_output):
        self.grad_weights = np.outer(grad_output, self.input)
        self.grad_biases = grad_output

        return self.weights.T @ grad_output

    def update(self, lr):
        self.weights -= lr * self.grad_weights
        self.biases -= lr * self.grad_biases
