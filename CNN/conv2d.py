"""
X = [[x_00, x_01, x_02],
     [x_10, x_11, x_12],
     [x_20, x_21, x_22]]

K = [[w_00, w_01],
     [w_10, w_11]]

grad_output = a  b
              c  d

Y = y_00 y_01
    y_10 y_11

y_00 = x_00 * w_00 + x_01 * w_01 + x_10 * w_10 + x_11 * w_11
y_01 = x_01 * w_00 + x_02 * w_01 + x_11 * w_10 + x_12 * w_11
y_10 = x_10 * w_00 + x_11 * w_01 + x_20 * w_10 + x_21 * w_11
y_11 = x_11 * w_00 + x_12 * w_01 + x_21 * w_10 + x_22 * w_11

∂L / ∂w_00 = a * x_00 + b * x_01 + c * x_10 + d * x_11
∂L / ∂w_01 = a * x_01 + b * x_02 + c * x_11 + d * x_12
∂L / ∂w_10 = a * x_10 + b * x_11 + c * x_20 + d * x_21
∂L / ∂w_11 = a * x_11 + b * x_12 + c * x_21 + d * x_22

--------------------------------------------------------

∂L / ∂x_00 = a * w_00
∂L / ∂x_01 = a * w_01 + b * w_00
∂L / ∂x_02 = b * w_01
∂L / ∂x_10 = a * w_10 + c * w_00
∂L / ∂x_11 = a * w_11 + b * w_10 + c * w_01 + d * w_00
∂L / ∂x_12 = b * w_11 + d * w_01
∂L / ∂x_20 = c * w_10
∂L / ∂x_21 = c * w_11 + d * w_10
∂L / ∂x_22 = d * w_11


fliplr =  w_01 w_00     flipud =  w_11 w_10
          w_11 w_10               w_01 w_00


grad_output_padded = 0  0  0  0
                     0  a  b  0
                     0  c  d  0
                     0  0  0  0

grad_input = a * w_00          a * w_01 + b * w_00                  b * w_01
             a*w_10 + c*w_00   a*w_11 + b*w_10 + c*w_01 + d*w_00  b*w_11 + d*w_01
             c*w_10            c*w_11 + d*w_10                       d*w_11


"""

import numpy as np


class Conv2d:
    def __init__(self, kernel_size=3):
        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.1

    def forward(self, x):
        self.input = x
        h, w = x.shape
        k_h, k_w = self.kernel.shape
        output_h = h - k_h + 1
        output_w = w - k_w + 1
        output = np.zeros((output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                region = x[i : i + k_h, j : j + k_w]
                output[i, j] = np.sum(region * self.kernel)

        return output

    def backward(self, grad_output):
        h, w = self.input.shape
        k_h, k_w = self.kernel.shape
        o_h, o_w = grad_output.shape

        self.grad_weights = np.zeros_like(self.kernel)

        for i in range(o_h):
            for j in range(o_w):
                patch = self.input[i : i + k_h, j : j + k_w]
                self.grad_weights += grad_output[i, j] * patch

        flipped = np.flipud(np.fliplr(self.kernel))
        padded_grad = np.pad(
            grad_output, ((k_h - 1, k_h - 1), (k_w - 1, k_w - 1)), "constant"
        )

        grad_input = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                region = padded_grad[i : i + k_h, j : j + k_w]
                grad_input[i, j] = np.sum(region * flipped)
        return grad_input

    def update(self, lr):
        self.kernel -= lr * self.grad_weights
