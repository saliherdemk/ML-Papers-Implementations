import numpy as np


class CrossEntropyLoss:
    def forward(self, logits, target):
        self.target = target
        # exps = np.exp(logits - np.max(logits))
        exps = np.exp(logits)
        self.probs = exps / np.sum(exps)
        return -np.log(self.probs[target])

    def backward(self):
        grad = self.probs.copy()
        grad[self.target] -= 1
        return grad


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.flatten()

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Relu:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
