# K-Means Clustering Algorithm

K-Means is an unsupervised machine learning algorithm used for clustering data into groups (clusters). The main goal is to divide a dataset into **K** distinct, non-overlapping clusters based on the similarity of the data points.

1. **Initialization**: Choose the number of clusters \( K \) and randomly initialize \( K \) cluster centroids.
2. **Assignment**: Assign each data point to the nearest cluster centroid.
3. **Update**: Recalculate the centroids as the mean of all data points in the cluster.
4. **Repeat**: Continue steps 2 and 3 until centroids do not change significantly or the maximum number of iterations is reached.

---

We will cluster the following 2D points into $K = 2$ clusters:

$$
(2, 3), \, (3, 3), \, (6, 8), \, (7, 9), \, (10, 12), \, (11, 14)
$$

Randomly choose two initial centroids:

$$
\mu_1 = (2, 3), \quad \mu_2 = (7, 9)
$$

---


For each point, calculate the squared Euclidean distance to both centroids and assign it to the nearest cluster.

For $(2, 3)$:

$$
\| (2, 3) - (2, 3) \|^2 = (2-2)^2 + (3-3)^2 = 0
$$

$$
\| (2, 3) - (7, 9) \|^2 = (2-7)^2 + (3-9)^2 = 5^2 + 6^2 = 25 + 36 = 61
$$

Since $0 < 61$, assign $(2, 3)$ to $\text{Cluster 1}$.

---

For $(3, 3)$:

$$
\| (3, 3) - (2, 3) \|^2 = (3-2)^2 + (3-3)^2 = 1
$$

$$
\| (3, 3) - (7, 9) \|^2 = (3-7)^2 + (3-9)^2 = 4^2 + 6^2 = 16 + 36 = 52
$$

Assign $(3, 3)$ to $\text{Cluster 1}$.

---
For $(6, 8)$:

$$
\| (6, 8) - (2, 3) \|^2 = (6-2)^2 + (8-3)^2 = 4^2 + 5^2 = 16 + 25 = 41
$$

$$
\| (6, 8) - (7, 9) \|^2 = (6-7)^2 + (8-9)^2 = 1^2 + 1^2 = 2
$$

Assign $(6, 8)$ to $\text{Cluster 2}$.

---
For $(7, 9)$:

$$
\| (7, 9) - (2, 3) \|^2 = (7-2)^2 + (9-3)^2 = 5^2 + 6^2 = 25 + 36 = 61
$$

$$
\| (7, 9) - (7, 9) \|^2 = (7-7)^2 + (9-9)^2 = 0
$$

Assign $(7, 9)$ to $\text{Cluster 2}$.

---
For $(10, 12)$:

$$
\| (10, 12) - (2, 3) \|^2 = (10-2)^2 + (12-3)^2 = 8^2 + 9^2 = 64 + 81 = 145
$$

$$
\| (10, 12) - (7, 9) \|^2 = (10-7)^2 + (12-9)^2 = 3^2 + 3^2 = 9 + 9 = 18
$$

Assign $(10, 12)$ to $\text{Cluster 2}$.

---
For $(11, 14)$:

$$
\| (11, 14) - (2, 3) \|^2 = (11-2)^2 + (14-3)^2 = 9^2 + 11^2 = 81 + 121 = 202
$$

$$
\| (11, 14) - (7, 9) \|^2 = (11-7)^2 + (14-9)^2 = 4^2 + 5^2 = 16 + 25 = 41
$$

Assign $(11, 14)$ to $\text{Cluster 2}$.

---

### Clusters After Assignment:
- **Cluster 1**: $(2, 3), (3, 3)$
- **Cluster 2**: $(6, 8), (7, 9), (10, 12), (11, 14)$

---

## Update Centroids

Recalculate the centroids as the mean of the points in each cluster.

For $\mu_1$ (Cluster 1):

$$
\mu_1 = \frac{(2, 3) + (3, 3)}{2} = \left( \frac{2+3}{2}, \frac{3+3}{2} \right) = (2.5, 3)
$$

For $\mu_2$ (Cluster 2):

$$
\mu_2 = \frac{(6, 8) + (7, 9) + (10, 12) + (11, 14)}{4} = \left( \frac{6+7+10+11}{4}, \frac{8+9+12+14}{4} \right)
$$

$$
\mu_2 = \left( \frac{34}{4}, \frac{43}{4} \right) = (8.5, 10.75)
$$

---

## Repeat

$$
\mu_1 = (2.5, 3), \quad \mu_2 = (8.5, 10.75)
$$


For $(2, 3)$:

$$
\| (2, 3) - (2.5, 3) \|^2  = 0.25
$$

$$
\| (2, 3) - (8.5, 10.75) \|^2= 102.3125
$$

Since $0.25 < 102.3125$, assign $(2, 3)$ to **Cluster 1**.

---

For $(3, 3)$:

$$
\| (3, 3) - (2.5, 3) \|^2 = 0.25
$$

$$
\| (3, 3) - (8.5, 10.75) \|^2 = 90.3125
$$

Since $0.25 < 90.3125$, assign $(3, 3)$ to **Cluster 1**.

---

For $(6, 8)$:

$$
\| (6, 8) - (2.5, 3) \|^2 = 37.25
$$

$$
\| (6, 8) - (8.5, 10.75) \|^2 = 13.8125
$$

Since $13.8125 < 37.25$, assign $(6, 8)$ to **Cluster 2**.

---

For $(7, 9)$:

$$
\| (7, 9) - (2.5, 3) \|^2 = 56.25
$$

$$
\| (7, 9) - (8.5, 10.75) \|^2 = 5.3125
$$

Since $5.3125 < 56.25$, assign $(7, 9)$ to **Cluster 2**.

---

For $(10, 12)$:

$$
\| (10, 12) - (2.5, 3) \|^2 = 137.25
$$

$$
\| (10, 12) - (8.5, 10.75) \|^2 = 3.8125
$$

Since $3.8125 < 137.25$, assign $(10, 12)$ to **Cluster 2**.

---

For $(11, 14)$:

$$
\| (11, 14) - (2.5, 3) \|^2 = 193.25
$$

$$
\| (11, 14) - (8.5, 10.75) \|^2 = 16.8125
$$

Since $16.8125 < 193.25$, assign $(11, 14)$ to **Cluster 2**.

---

### Clusters After Assignment:
- **Cluster 1**: $(2, 3), (3, 3)$
- **Cluster 2**: $(6, 8), (7, 9), (10, 12), (11, 14)$

Since the clusters did not change after this iteration, the algorithm has converged.

<!-- <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script> -->

