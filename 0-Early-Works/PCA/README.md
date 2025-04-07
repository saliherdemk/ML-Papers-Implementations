### Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique used to reduce the number of features in a dataset while preserving as much variance as possible.

It has many prerequisites in order to fully understand PCA, so this will only cover the methodology.

---


| Systolic BP (sbp) | Diastolic BP (dbp)|
|-------------|--------------|
| 126         | 78           |
| 128         | 80           |
| 128         | 82           |
| 130         | 82           |
| 130         | 84           |
| 132         | 86           |

### Center Data

$$
\mu_{sbp} = \frac{126 + 128 +128 + 130 + 130 + 132}{6} = 129
$$

$$
\mu_{dbp} = \frac{78 + 80 + 82 + 82 + 84 + 86}{6} = 82
$$

| centered sbp | centered dbp|
|-------------|--------------|
| 126 - 129 = -3 | 78 - 82 = -4 |
| 238 - 129 = -1 | 80 - 82 = -2 |
| 128 - 129 = -1| 82 - 82 = 0 |
| 130 - 129 = 1 | 82 - 82 = 0 |
| 130 - 129 = 1 | 84 - 82 = 2|
| 132 - 129 = 3| 86 - 82 = 4|

### Covariance Matrix

Our means ($\overline{csbp}$, $\overline{cdbp}$) are 0 since we centered data.

$$
var(csbp) = \frac{1}{n-1}\sum_{i=1}^{n}(csbp_i - \overline{csbp})^2 = 4.4
$$

$$
var(cdbp) = \frac{1}{n-1}\sum_{i=1}^{n}(cdbp_i - \overline{cdbp})^2 = 8
$$

$$
cov(csbp, cdbp) = cov(cdbp, csbp) = \frac{1}{n-1}\sum_{i=1}^{n}(csbp_i - \overline{csbp}) *(cdbp_i - \overline{cdbp}) = 5.6
$$

| | sbp | dbp|
|--- |---|---|
| sbp | 4.4 |5.6 |
| dbp | 5.6 | 8 |

### Eigenvalues

$$
det|A - \lambda I| = 0
$$

$$ 
det(\begin{bmatrix} 4.4 & 5.6 \\ 5.6 & 8 \\ \end{bmatrix} - \begin{bmatrix} \lambda & 0 \\ 0 & \lambda \end{bmatrix}) = 0
$$

$$
(4.4 - \lambda) (8 - \lambda) - 5.6 * 5.6 = 0
$$

$$
\lambda^2 - 12.4\lambda + 3.84 = 0
$$

$$
\lambda_1 = 0.32 \quad \lambda_2 = 12.08
$$

### Eigenvectors

$$
A \cdot v = \lambda \cdot v
$$

$$
\begin{bmatrix} 
4.4 & 5.6 \\ 
5.6 & 8 
\end{bmatrix} 
\cdot
\begin{bmatrix} 
x \\
y 
\end{bmatrix} = 12.08 
\cdot 
\begin{bmatrix} x \\
y \end{bmatrix}
$$

$$
v_2 = \begin{bmatrix} 1 \\
1.37 \end{bmatrix}
$$

L2 normalization

$$
||v_2|| = \sqrt{1^2 + 1.37^2} \approx 1.696
$$

$$
\hat{v_2} = \frac{1}{1.696} \begin{bmatrix} 1 \\
1.37 \end{bmatrix} = \begin{bmatrix} 0.59 \\
0.81 \end{bmatrix}
$$

So

$$
\lambda_1 = 0.32 \quad v_1 =\begin{bmatrix} -0.81 \\ 
0.59 \end{bmatrix}
$$

$$
\lambda_2 = 12.08 \quad v_1 =\begin{bmatrix} 0.59 \\
0.81 \end{bmatrix}
$$

### Order and Merge

$$
V = \begin{bmatrix} 0.59 & -0.81 \\
0.81 & 0.59 \end{bmatrix}
$$

The first column corresponds to the highest eigenvalue ($\lambda_2 > \lambda_1$)

### Calculate Principle Components


$$
V = \begin{bmatrix} 0.59 & -0.81 \\
0.81 & 0.59 \end{bmatrix} \quad
D = \begin{bmatrix} -3 & -4 \\
-1 & -2 \\
-1 & 0 \\
1 & 0 \\
1 & 2 \\
3 & 4
\end{bmatrix}
$$


$$ 
DV = \begin{bmatrix} -3 & -4 \\
-1 & -2 \\
-1 & 0 \\
1 & 0 \\
1 & 2 \\
3 & 4 
\end{bmatrix}
\begin{bmatrix} 0.59 & -0.81 \\
0.81 & 0.59 
\end{bmatrix} = \begin{bmatrix} -5.0 & 0.1 \\
-2.2 & -0.4 \\
-0.6 & 0.8 \\
0.6 & -0.8 \\
2.2 & 0.4 \\
5.0 & -0.1
\end{bmatrix}
$$


| PC1  | PC2  |
|------|------|
| -5.0 | 0.1  |
| -2.2 | -0.4 |
| -0.6 | 0.8  |
| 0.6  | -0.8 |
| 2.2  | 0.4  |
| 5.0  | -0.1 |
|var = 12.08 | var = 0.32|

Covariance matrix of the principal components

| | pc1 | pc2|
|--- |---|---|
| pc1 | 12.08 | 0 |
| pc2 | 0 | 0.32 |

which means pc1 and pc2 have no correlation with each other.

$$
pc_1 = 0.59 \cdot csbp + 0.81 \cdot cdbp
$$

$$
pc_2 = -0.81 \cdot csbp + 0.59 \cdot cdbp
$$

### Data Dimensionality Reduction

$$
var(pc_1) = 12.08 \quad var(pc_2)=0.32
$$

$$
\frac{12.08}{12.08 + 0.32} \approx 97 \%
$$

Since most of the variance kept in first principle component, we can ignore $pc_2$ and continue with pc_1


## References

- https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab **
- https://www.youtube.com/watch?v=FgakZw6K1QQ  
- https://www.youtube.com/watch?v=mxkGMbrobY0
- https://www.youtube.com/watch?v=S51bTyIwxFs **
- https://www.youtube.com/watch?v=nbBvuuNVfco
