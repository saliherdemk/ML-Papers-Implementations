
# Linear Regression Algorithm

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable $y$ and one or more independent variables $X$. The objective is to fit a line (or hyperplane in higher dimensions) that minimizes the error between the predicted and actual values.

---

The hypothesis function for linear regression is:

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

Where:
- $h_\theta(x)$ is the predicted value.
- $x_i$ are the input features.
- $\theta_i$ are the parameters (weights) of the model.

In matrix form:

$$
h_\theta(X) = X \theta
$$

Where:
- $X$ is the design matrix containing the input features.
- $\theta$ is the parameter vector.

---

## Cost Function

The cost function for linear regression is the Mean Squared Error (MSE):

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^m \left( y^{(i)} - h_\theta(x^{(i)}) \right)^2
$$

Where:
- $m$ is the number of training examples.
- $h_\theta(x^{(i)})$ is the predicted value for the $i$-th example.
- $y^{(i)}$ is the actual value for the $i$-th example.

The goal is to minimize $J(\theta)$.

---

## Gradient Descent Algorithm

To minimize the cost function, we use the gradient descent algorithm. The parameters $\theta$ are updated iteratively as follows:

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

Where:
- $\alpha$ is the learning rate.
- $\frac{\partial J(\theta)}{\partial \theta_j}$ is the partial derivative of the cost function with respect to $\theta_j$.


---
In my implementation, $\theta$ will consist of weights and biases, so we need to take partial derivatives separately for each of these parameters


$$
h(x) = wx + b
$$

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} -(w^ix^i + b))^2
$$

$$
\frac{\partial J(\theta)}{\partial w} =  -\frac{2}{m} \sum_{i=1}^{m} (y^{(i)} - (w^ix^i + b) ) x^i
$$

$$
\frac{\partial J(\theta)}{\partial b} =  -\frac{2}{m} \sum_{i=1}^{m} (y^{(i)} -(w^ix^i + b) )
$$
