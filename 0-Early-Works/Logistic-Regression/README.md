# Logistic Regression Algorithm

Logistic Regression is a supervised learning algorithm used for binary classification. The main goal is to predict the probability that a given input belongs to a particular class.

---


Logistic regression predicts the probability of a class label $y \in \{0, 1\}$ using the **logistic (sigmoid)** function:  

$$
h_\theta(x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n
$$

Here:
- $\sigma(z)$: Sigmoid function maps any real number to a value between 0 and 1.
- $z$: Linear combination of the input features and parameters $\theta$.

---

## Cost Function

The cost function for logistic regression is derived from a probabilistic approach. The model predicts 

$P(y = 1 \mid x; \theta) = h_\theta(x)$ and $P(y = 0 \mid x; \theta) = 1 - h_\theta(x)$.

The likelihood for a single example is:  

$$
P(y \mid x; \theta) = h_\theta(x)^y (1 - h_\theta(x))^{1 - y}
$$

The log-likelihood for the entire dataset is:  

$$
\ell(\theta) = \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

The negative log-likelihood (minimized in optimization) becomes the **cross-entropy loss**:  

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

Where:
- $m$: Number of training examples.
- $y^{(i)}$: True label for the $i$-th example ($y^{(i)} \in \{0, 1\}$).
- $h_\theta(x^{(i)})$: Predicted probability for the $i$-th example.

This cost function ensures that the model outputs probabilities that closely align with the true labels.

---
## 3. Optimization

To find the optimal parameters $\theta$, we maximize the log-likelihood function $\ell(\theta)$, which is equivalent to minimizing the **negative log-likelihood** (cross-entropy loss) function $J(\theta)$. The goal is to adjust $\theta$ such that the model's predicted probabilities match the true labels as closely as possible.

### Log-Likelihood Function

The log-likelihood for the entire dataset is:

$$
\ell(\theta) = \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

Where:
- $m$: Number of training examples.
- $y^{(i)}$: True label for the $i$-th example ($y^{(i)} \in \{0, 1\}$).
- $h_\theta(x^{(i)})$: Predicted probability for the $i$-th example.

### Negative Log-Likelihood (Cost Function)

The negative log-likelihood, or the **cross-entropy loss** function, is minimized in the optimization process:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

## Derivation of the Gradient for Logistic Regression


$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$


$$
\frac{\partial J(\theta)}{\partial \theta_j} = ?
$$

---


$$
\frac{\partial}{\partial \theta_j} \left[ y^{(i)} \log(h_\theta(x^{(i)})) \right] = y^{(i)} \cdot \frac{\partial}{\partial \theta_j} \log(h_\theta(x^{(i)}))
$$


$$
\frac{\partial}{\partial \theta_j} \log(h_\theta(x^{(i)})) = \frac{1}{h_\theta(x^{(i)})} \cdot \frac{\partial h_\theta(x^{(i)})}{\partial \theta_j}
$$

$$
h_\theta(x^{(i)}) = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}, z^{(i)} = \sum_{k=0}^n \theta_k x_k^{(i)} = \theta^T x^{(i)}
$$

$$
\frac{\partial h_\theta(x^{(i)})}{\partial \theta_j} = \frac{\partial \sigma(z^{(i)})}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial \theta_j}
$$

---
$$
\sigma(z) = (1 + e^{-z})^{-1}
$$

$$
\frac{\partial \sigma(z)}{\partial z} = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{\partial}{\partial z}(1 + e^{-z})
$$

$$
\frac{\partial}{\partial z}(1 + e^{-z}) = -e^{-z}
$$

$$
\frac{\partial \sigma(z)}{\partial z} = \frac{e^{-z}}{(1 + e^{-z})^2}
$$

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad 1 - \sigma(z) = \frac{e^{-z}}{1 + e^{-z}}
$$

$$
\frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) \cdot (1 - \sigma(z))
$$

$$
\frac{\partial \sigma(z)}{\partial z} = \sigma(z) \cdot (1 - \sigma(z))
$$

---


$$
\frac{\partial z^{(i)}}{\partial \theta_j} = x_j^{(i)}
$$

$$
\frac{\partial h_\theta(x^{(i)})}{\partial \theta_j} = h_\theta(x^{(i)}) \cdot (1 - h_\theta(x^{(i)})) \cdot x_j^{(i)}
$$

$$
\frac{\partial}{\partial \theta_j} \log(h_\theta(x^{(i)})) = \frac{1}{h_\theta(x^{(i)})} \cdot h_\theta(x^{(i)}) \cdot (1 - h_\theta(x^{(i)})) \cdot x_j^{(i)}
$$

$$
\frac{\partial}{\partial \theta_j} \log(h_\theta(x^{(i)})) = (1 - h_\theta(x^{(i)})) \cdot x_j^{(i)}
$$

$$
\frac{\partial}{\partial \theta_j} \left[ y^{(i)} \log(h_\theta(x^{(i)})) \right] = y^{(i)} \cdot (1 - h_\theta(x^{(i)})) \cdot x_j^{(i)}
$$

---

Same thing

$$
\frac{\partial}{\partial \theta_j} \left[ (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] = -(1 - y^{(i)}) \cdot h_\theta(x^{(i)}) \cdot x_j^{(i)}
$$

---


$$
\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \cdot (1 - h_\theta(x^{(i)})) \cdot x_j^{(i)} - (1 - y^{(i)}) \cdot h_\theta(x^{(i)}) \cdot x_j^{(i)} \right]
$$


$$
\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m} \sum_{i=1}^m x_j^{(i)} \left[ y^{(i)} (1 - h_\theta(x^{(i)})) - (1 - y^{(i)}) h_\theta(x^{(i)}) \right]
$$


$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left[ h_\theta(x^{(i)}) - y^{(i)} \right] x_j^{(i)}
$$

---

### Update 

$$
Q_j := Q_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

## References

- https://www.youtube.com/watch?v=het9HFqo1TQ
