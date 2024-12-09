import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self, learning_rate=0.2, epochs=20) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            total_error = 0
            for i in range(len(x)):
                predicted = self.forward(x[i])
                target = y[i]
                diff = target - predicted

                self.weights += self.lr * diff * x[i]
                self.bias += self.lr * diff

                total_error += diff**2

            self.errors.append(total_error / len(x))

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

model = Adaline()
model.fit(X, Y)


print(0, 0, "=>", model.forward([0, 0]))
print(0, 1, "=>", model.forward([0, 1]))
print(1, 0, "=>", model.forward([1, 0]))
print(1, 1, "=>", model.forward([1, 1]))


plt.plot(range(1, model.epochs + 1), model.errors, marker="o")
plt.title("Training Error Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()
