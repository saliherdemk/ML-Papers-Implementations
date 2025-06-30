"""
Pooling
---
X= 1  5  4
   9  2  1
   6  3  2

kernel_size = 2,2

Max Pooling
--
Y = 9 5
    9 3

|grad_output| = a b
                c d

for
1  5
9  2      we have a

for second window b, third c, fourth d
So,
∂L / ∂X = 0     b  0
          a + c 0  0
          0     d  0

Avg Pooling
--
Y = 4.25 3
    5 2

for avg we have to distribute because each value contributes equally

∂L / ∂X = a/4          a/4 + b/4                b/4
          a/4 + c/4    a/4 + b/4 + c/4 + d/4    b/4 + d/4
          c/4          c/4 + d/4                d/4

"""

import numpy as np


class Pooling:
    def __init__(self, pool_type="max", kernel_size=(3, 3)):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.indicies = []

    def forward(self, x):
        self.input_shape = x.shape
        h, w = x.shape
        kh, kw = self.kernel_size
        output_h = h - kh + 1
        output_w = w - kw + 1
        output = np.zeros((output_h, output_w))

        self.indices = []

        for i in range(output_h):
            for j in range(output_w):
                region = x[i : i + kh, j : j + kw]
                if self.pool_type == "max":
                    output[i, j] = np.max(region)
                    max_index = np.unravel_index(np.argmax(region), region.shape)
                    absolute_index = (i + max_index[0], j + max_index[1])
                    self.indices.append(absolute_index)
                elif self.pool_type == "avg":
                    output[i, j] = np.mean(region)

        return output

    def backward(self, grad_output):
        grad_input = np.zeros(self.input_shape)
        kh, kw = self.kernel_size
        output_h, output_w = grad_output.shape

        if self.pool_type == "max":
            for idx, (x_i, x_j) in enumerate(self.indices):
                i = idx // output_w
                j = idx % output_w
                grad_input[x_i, x_j] += grad_output[i, j]

        elif self.pool_type == "avg":
            for i in range(output_h):
                for j in range(output_w):
                    grad_input[i : i + kh, j : j + kw] += grad_output[i, j] / (kh * kw)
        return grad_input
