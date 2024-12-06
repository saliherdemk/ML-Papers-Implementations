# The Perceptron / Principles Of Neurodynamics 

**Frank Rosenblatt - 1958 / 1962**

It's really hard to pull the exact formula from the book but standart learning rule formula as follows

$$
\mathbf{w} \leftarrow \mathbf{w} + \Delta \mathbf{w}
$$

where the weight update $\Delta \mathbf{w}$ is given by:

$$
\Delta \mathbf{w} = \eta (y - \hat{y}) \mathbf{x}
$$

where:
- $\mathbf{w}$ is the vector of weights,
- $\mathbf{x}$ is the input vector,
- $\hat{y}$ is the predicted output,
- $y$ is the true output,
- $\eta$ is the learning rate.

And the perceptron uses a step activation function:

$$
\hat{y} = 
\begin{cases} 
1 & \text{if } \mathbf{w} \cdot \mathbf{x} \geq \theta \\
0 & \text{otherwise}
\end{cases}
$$

where $\theta$ is the threshold value.

## References

- https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf 
- https://archive.org/details/principles-of-neurodynamics/page/n311/mode/2up?view=theater
- https://www.youtube.com/watch?v=9ttOIR0vG-4

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

