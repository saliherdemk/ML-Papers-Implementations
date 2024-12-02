import numpy as np


class Hebbian:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size, output_size)
        self.learning_rate = learning_rate

    def train(self, inputs, outputs):
        for i in range(len(inputs)):
            input_vector = inputs[i]
            output_vector = outputs[i]
            self.weights += self.learning_rate * np.outer(input_vector, output_vector)

    def predict(self, input_vector):
        raw_output = np.dot(input_vector, self.weights)
        return (raw_output >= 1).astype(int)


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [0], [0], [1]])

network = Hebbian(input_size=2, output_size=1)

network.train(inputs, outputs)

for test_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    predicted_output = network.predict(test_input)
    print(f"Predicted output for input {test_input}: {predicted_output}")
