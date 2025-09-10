[Original Paper: XGBoost: A Scalable Tree-based Boosting System](https://arxiv.org/abs/1603.02754)

# Gradient Boosting

### Core Concept

Gradient boosting builds models sequentially, where each new model corrects the mistakes of all previous models. Instead of training models independently, each model learns from the residual errors of the ensemble.

The key insight: if we have a prediction that's "close but not perfect," we can train another model to predict the error, then add that correction to our original prediction.

Gradient boosting builds an additive model:

$$
F_M(x) = \sum_{m=1}^M \gamma_m h_m(x)
$$

Where
- $F_M(x)$: The final prediction of the ensemble after $M$ steps.
- $h_m(x)$: The base learner (usually a decision tree) trained at step $m$.
- $\gamma_m$: A weight or step size for that learner.

The idea: instead of making one big model, we build many small models and add their contributions together. Each new model fixes mistakes of the previous models.

Think of it like iteratively improving your prediction:
- Make a guess.
- See how wrong it is.
- Train a new model to correct that wrong part.
- Add that correction to your guess.

We define a differentiable loss function:

$$
L(y, F(x))
$$

- $y$: actual value.
- $F(x)$: current prediction.

At iteration $m$, compute the negative gradient (pseudo-residuals):

$$
r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$


It tells us how much we should change our prediction for sample $i$ to reduce the loss. For our example, L is mean squares error loss

$$
L(y, F(x)) = \frac{1}{2}(y - F(x))^2
$$

Take derivative w.r.t. predictions F(x):

$$
\frac{\partial L}{\partial F(x)} = F(x) - y
$$

So the negative gradient is

$$
r = y - F(x)
$$

We call this residuals.

Base learner is usually a tree. Here is a toy example to see how full pipeline works.

| Hours Studied | Sleep Hours | Score |
|---------------|-------------|-------|
| 2.0           | 6.0         | 50.0  |
| 4.0           | 5.0         | 60.0  |
| 1.0           | 8.0         | 45.0  |
| 5.0           | 5.0         | 65.0  |
| 3.0           | 7.0         | 55.0  |
| 6.0           | 4.0         | 70.0  |
| 2.0           | 7.0         | 52.0  |
| 7.0           | 3.0         | 75.0  |
| 3.0           | 6.0         | 58.0  |
| 5.0           | 6.0         | 68.0  |


First, we need an first residual usually considered the mean of the label.

$$
F_0= \frac{50 + 60 + 45 + 65 + 55 + 70 + 52 + 75 + 58 + 68}{10} = 59.8
$$

And the residuals are just $y - F_0$

| Hours Studied | Sleep Hours | Score | Baseline (F₀ = 59.8) | Residual (y - F₀) |
|---------------|-------------|-------|-----------------------|-------------------|
| 2.0           | 6.0         | 50.0  | 59.8                 | -9.8              |
| 4.0           | 5.0         | 60.0  | 59.8                 | +0.2              |
| 1.0           | 8.0         | 45.0  | 59.8                 | -14.8             |
| 5.0           | 5.0         | 65.0  | 59.8                 | +5.2              |
| 3.0           | 7.0         | 55.0  | 59.8                 | -4.8              |
| 6.0           | 4.0         | 70.0  | 59.8                 | +10.2             |
| 2.0           | 7.0         | 52.0  | 59.8                 | -7.8              |
| 7.0           | 3.0         | 75.0  | 59.8                 | +15.2             |
| 3.0           | 6.0         | 58.0  | 59.8                 | -1.8              |
| 5.0           | 6.0         | 68.0  | 59.8                 | +8.2              |

Then we'll pass the all the features and the corresponding residuals to constructing the first tree.

| Hours Studied | Sleep Hours |  Residual (y - F₀) |
|---------------|-------------| -----------------|
| 2.0           | 6.0         | -9.8             |
| 4.0           | 5.0         | +0.2             |
| 1.0           | 8.0         | -14.8            |
| 5.0           | 5.0         | +5.2             |
| 3.0           | 7.0         | -4.8             |
| 6.0           | 4.0         | +10.2            |
| 2.0           | 7.0         | -7.8             |
| 7.0           | 3.0         | +15.2            |
| 3.0           | 6.0         | -1.8             |
| 5.0           | 6.0         | +8.2             |

Now as an example, I'll walk through the simplest version of the split method, DecisionStump. For each feature we'll calculate the total error for that split and choose the minimum feature.

#### Feature-0 Hours Studied

We have 7 unique values. [1,2,3,4,5,6,7]

For threshold = 1, we have [-14.8] on left and the rest on right.
Left mean is -14.8. Right mean is 1.64.

```
                 [Hours Studied ≤ 1?]
                 /                 \
          Yes (-14.8)             No (1.76)
```

Then we're calculating the error for this tree.

| Hours Studied | Residual (y − F₀) | Prediction | Error = (Residual − Prediction)² |
| ------------- | ----------------- | ---------- | -------------------------------- |
| 1.0           | -14.8             | -14.8      | 0.00                             |
| 2.0           | -9.8              | 1.64       | 130.98                           |
| 4.0           | 0.2               | 1.64       | 2.09                             |
| 5.0           | 5.2               | 1.64       | 12.64                            |
| 3.0           | -4.8              | 1.64       | 41.53                            |
| 6.0           | 10.2              | 1.64       | 73.20                            |
| 2.0           | -7.8              | 1.64       | 89.20                            |
| 7.0           | 15.2              | 1.64       | 183.75                           |
| 3.0           | -1.8              | 1.64       | 11.86                            |
| 5.0           | 8.2               | 1.64       | 42.98                            |

Total error $\approx$ 588.22 for feature-0. Then we will calculate total error for each candidate thresholds as well. Here is the all errors for feature-0 (`exercise/main.ipynb`):

| Threshold | Total SSE |
| --------- | --------- |
| 1.0       | 588.22    |
| 2.0       | 331.71    |
| 3.0       | 223.20    |
| 4.0       | 204.33    |
| 5.0       | 428.38    |
| 6.0       | 574.89    |

Then we'll calculate the exam same thing for feature-1. Here is the feature-1 errors:

| Threshold | Total SSE |
| --------- | --------- |
| 3.0       | 574.89    |
| 4.0       | 428.38    |
| 5.0       | 436.33    |
| 6.0       | 474.10    |
| 7.0       | 588.22    |

Among all of the candidates, the minimum error is belongs to threshold value 4.0. So we'll split the tree using that.

```
                 [Hours Studied ≤ 4?]
                 /                 \
          Yes (-6.46)             No (9.7)
```

Then we'll get the predictions from that tree and update our initial prediction. For learning_rate is 0.1, here is the updating process:

$$
F_1 = F_0 + lr \cdot treeOutput , lr=0.1
$$

| Hours Studied | Sleep Hours | Baseline (F₀) | Tree Output | Updated Baseline (F₁) |
|---------------|-------------|----------------|-------------|-----------------------|
| 2.0           | 6.0         | 59.8           | -6.47       | 59.15                 |
| 4.0           | 5.0         | 59.8           | -6.47       | 59.15                 |
| 1.0           | 8.0         | 59.8           | -6.47       | 59.15                 |
| 5.0           | 5.0         | 59.8           | 9.70        | 60.77                 |
| 3.0           | 7.0         | 59.8           | -6.47       | 59.15                 |
| 6.0           | 4.0         | 59.8           | 9.70        | 60.77                 |
| 2.0           | 7.0         | 59.8           | -6.47       | 59.15                 |
| 7.0           | 3.0         | 59.8           | 9.70        | 60.77                 |
| 3.0           | 6.0         | 59.8           | -6.47       | 59.15                 |
| 5.0           | 6.0         | 59.8           | 9.70        | 60.77                 |

Then we'll use the $F_1$ to calculate the residuals for the next tree and keep going like that. After 200 iteration, here is the result:

| Hours Studied | Sleep Hours | Predicted Score | Actual Score |
|---------------|-------------|----------------|--------------|
| 2.0           | 6.0         | 51.00          | 50.0         |
| 4.0           | 5.0         | 60.00          | 60.0         |
| 1.0           | 8.0         | 45.00          | 45.0         |
| 5.0           | 5.0         | 66.50          | 65.0         |
| 3.0           | 7.0         | 56.50          | 55.0         |
| 6.0           | 4.0         | 70.00          | 70.0         |
| 2.0           | 7.0         | 51.00          | 52.0         |
| 7.0           | 3.0         | 75.00          | 75.0         |
| 3.0           | 6.0         | 56.50          | 58.0         |
| 5.0           | 6.0         | 66.50          | 68.0         |


