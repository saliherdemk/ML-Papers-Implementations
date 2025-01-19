# K-Nearest Neighbors (KNN) Algorithm

The basic idea is to predict the label of a given data point based on the labels of its closest neighbors in the feature space.


**Compute the distance between the data points**:
The distance between the data point for which we want to make a prediction and the training points is calculated.

Euclidean distance

$$ d(x_i, x_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2} $$

Manhattan distance - L1 norm

$$ d(x_i, x_j) = \sum_{k=1}^{n} |x_{ik} - x_{jk}| $$

Minkowski distance

The Minkowski distance is a generalization of both Euclidean and Manhattan distances. It is defined as:

$$
d(x_i, x_j) = \left( \sum_{k=1}^{n} |x_{ik} - x_{jk}|^p \right)^{1/p}
$$

- For $p=1$, this becomes the **Manhattan distance**.
- For $p=2$, this becomes the **Euclidean distance**.

where:
- $\mathbf{x}_i$ and $\mathbf{x}_j$ are the feature vectors of data points $i$ and $j$.
- $n$ is the number of features.
- $p$ is the order of the norm.

**Find the K nearest neighbors**:
Sort all training points based on the distance to the given point and select the top $K$ nearest neighbors.

**Prediction**:
- For **classification**, the prediction is based on the majority class of the $K$ nearest neighbors. If the most frequent class among the neighbors is $C$, the prediction for the input data point is $C$.
   
- For **regression**, the prediction is the average (or weighted average) of the target values of the $K$ nearest neighbors.

$$
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i
$$

where:
- $\hat{y}$ is the predicted value.
- $y_i$ is the target value of the $i$-th nearest neighbor.

