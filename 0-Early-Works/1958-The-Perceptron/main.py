import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.threshold = 0
        self.act_func = lambda x: np.where(x > self.threshold, 1, 0)
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i, _x in enumerate(x):
                predicted = self.forward(_x)
                delta = self.lr * (y[i] - predicted)
                self.weights += delta * _x
                self.bias += delta

    def forward(self, x):
        return self.act_func(np.matmul(self.weights.T, x) + self.bias)


p = Perceptron()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

p.fit(X, Y)

print(0, 0, "=>", p.forward([0, 0]))
print(0, 1, "=>", p.forward([0, 1]))
print(1, 0, "=>", p.forward([1, 0]))
print(1, 1, "=>", p.forward([1, 1]))
