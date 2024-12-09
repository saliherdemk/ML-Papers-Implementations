# Adaline - Madaline

**Bernand Widrow & Marcian Hoff - 1960**

ADALINE (Adaptive Linear Neuron or later Adaptive Linear Element) is an early single-layer artificial neural network and the name of the physical device that implemented it. It was developed by professor Bernard Widrow and his doctoral student Marcian Hoff at Stanford University in 1960. It is based on the perceptron and consists of weights, a bias, and a summation function. The weights and biases were implemented by rheostats (as seen in the "knobby ADALINE"), and later, memistors.

The difference between Adaline and the standard (Rosenblatt) perceptron is in how they learn. Adaline unit weights are adjusted to match a teacher signal, before applying the Heaviside function (see figure), but the standard perceptron unit weights are adjusted to match the correct output, after applying the Heaviside function.

A multilayer network of ADALINE units is known as a MADALINE.

Adaline is a single-layer neural network with multiple nodes, where each node accepts multiple inputs and generates one output. Given the following variables:

$$
y = \sum_{j=1}^n x_jw_j + \theta
$$

where

- $x$, the input vector
- $w$, the weight vector
- $n$, the number of inputs
- $\theta$, some constant
- $y$, the output of the model


Learning rule:

$$
w \leftarrow w + \eta (o-y)x
$$

where

- $\eta$, the learning rate
- $y$, the model output
- $o$, the target output
- $E = (o−y)^2$, the square of the error,

This update rule minimizes $E$, the square of the error, and is in fact the stochastic gradient descent update for linear regression.

## References
- https://en.wikipedia.org/wiki/ADALINE

