### Recurrent Neural Networks

Recurrent neural networks are networks with loops in them, allowing information to persist.


An RNN processes one element of a sequence at a time while maintaining a hidden state $h_t$ that captures information about the previous elements.

At timestep $t$:

- Input: $x_t$  
- Previous hidden state: $h_{t-1}$  
- Output: $y_t$  

The hidden state update and output are given by:

$$
h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

- $W_{hx}, W_{hh}, W_{hy}$: weight matrices  
- $b_h, b_y$: biases  

---


<center>
<img src="./media/rnn.png"> </img>
</center>

Each timestep shares the same parameters.

### Example

```
sequences = [
    ['a', 'b', 'EOS'],
    ['a', 'a', 'b', 'b', 'EOS'],
    ['a', 'a', 'a', 'b', 'b', 'b', 'EOS'],
    ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'EOS'],
    ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'EOS']
]
```

Apply one hot encoding. We had `UNK` token for unknown chars. With that, our vocabulary size is 4.

```
a: [1. 0. 0. 0.]
b: [0. 1. 0. 0.]
EOS: [0. 0. 1. 0.]
UNK: [0. 0. 0. 1.]
```

### Parameter shapes

- $W_{xh}$ shape = hidden\_size, vocab\_size  
- $W_{hh}$ shape = hidden\_size, hidden\_size  
- $b_h$ shape = hidden\_size
- $W_{hy}$ shape = vocab\_size, hidden\_size  
- $b_y$ shape = vocab\_size

The forward equations:

$$
h_t = \tanh(W_{xh} \cdot x + W_{hh} \cdot h_t + b_h)
$$

$$
y_t = W_{hy} \cdot h_t + b_y
$$

At each time step, we get output $y_t$ and hidden state $h_t$.

- $h_t$ is a summary of the past, a vector that contains compressed information about everything the RNN has seen up to time step $t$.
- $y_t$, called logits, represents raw predictions before softmax. These are not probabilities yet.

Each $y_t$ (logits) has shape vocab\_size, which in our case is 4 for the characters ['a', 'b', 'EOS', 'UNK'].

---

### Example

Input character $a$:

$$
a = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

Model output logits $y$:

$$
y = \begin{bmatrix} 4.8925 \times 10^{-4} \\ -7.6466 \times 10^{-5} \\ -2.9858 \times 10^{-5} \\ -3.3591 \times 10^{-4} \end{bmatrix}
$$

---

Cross-entropy loss calculation for 1-character prediction


$$
\text{logits} = \begin{bmatrix} -1.2366 \times 10^{-4} & 1.7240 \times 10^{-4} \end{bmatrix}
$$


$$
\text{targets} = \begin{bmatrix} 0 \end{bmatrix}
$$


$$
logsoftmax(x_i) = x_i - \log\left( \sum_j e^{x_j} \right)
$$


$$
\sum_j e^{x_j} = e^{-1.2366 \times 10^{-4}} + e^{1.7240 \times 10^{-4}} \approx 2
$$


$$
logsoftmax(x_0) = x_0 - \log\left( \sum_j e^{x_j} \right) = -1.2366 \times 10^{-4} - \log(2) \approx -0.6933
$$


$$
\text{loss} = - logsoftmax(x_0) \approx 0.6933
$$

---

If we have more than one character, the total loss is the mean of individual losses:

$$
\text{loss} = \frac{1}{N} \sum_{i=1}^{N} \text{loss}_i
$$



---

### Backpropagation Through Time 

To train RNNs, we use Backpropagation Through Time, a variant of backpropagation applied to the unrolled RNN across multiple timesteps. The gradients are computed across the entire sequence, allowing the model to learn temporal dependencies. 

However, RNNs suffer from vanishing gradients and exploding gradients problems especially when processing long sequences.

### Long Short Term Memory

