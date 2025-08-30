

[Original Paper: Attention is All You Need](https://arxiv.org/abs/1706.03762)

# Transformer Architecture

### Example Data

| x          | y                  |
|------------|------------------|
| 1845-01-05 | January 5, 1845   |
| 1467-07-28 | July 28, 1467     |
| 1468-01-11 | January 11, 1468  |
| 1996-09-08 | September 8, 1996 |
| 1959-03-02 | March 2, 1959     |

For this problem, let's say we'll match each character with a token. With that, our vocabulary looks like this:

- 0-9 -> 10 characters
- A-Z -> 26 characters
- a-z -> 26 characters
- Special characters: `-`, `,`, space, `<sos>`, `<eos>`, `<pad>` -> 6 characters

Our vocabulary size is 68. Then we're matching each token with a single number. For example, if we match all of them with their indexes, it will look like this:

| char      | token |
|-----------|-------|
| 0-9       | 0-9   |
| A-Z       | 10-35 |
| a-z       | 36-61 |
| -         | 62    |
| ,         | 63    |
| space     | 64    |
| `<sos>`   | 65    |
| `<eos>`   | 66    |
| `<pad>`   | 67    |

Here is the tokenization process:

`1676-11-30` -> `<sos>1676-11-30<eos>` -> `[65, 1, 6, 7, 6, 62, 1, 1, 62, 3, 0, 66]`

Max input length is 10 (`1676-11-30`) + 2 (`<sos><eos>`) = 12 and max output length is 18 (`September 28, 1976`) + 2 (`<sos><eos>`) = 20. 

We're padding to achive fixed input and output length.

`November 30, 1676` -> `<sos>November 30, 1676<eos><pad>` -> `[65, 23, 50, 57, 40, 48, 37, 40, 53, 64, 3, 0, 63, 64, 1, 6, 7, 6, 66, 67]`

### Token Embeddings

For neural networks, numeric IDs alone are not enough. A single number cannot capture relationships or contextual meaning between tokens. Embeddings provide the semantic and contextual depth necessary for neural networks to generate meaningful representations and make predictions.

For a vocabulary of size 68 and an embedding dimension of 16, we are creating a 16-dimensional embedding vector for each possible character in the vocabulary. Our sampled embedding vectors for token IDs `[65, 1, 3, 66]` look like this:


|id| Dim 1   | Dim 2   | Dim 3   | Dim 4   | Dim 5   | Dim 6   | Dim 7   | Dim 8   | Dim 9   | Dim 10  | Dim 11  | Dim 12  | Dim 13  | Dim 14  | Dim 15  | Dim 16  |
|---|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|65| 0.7886  | 0.7564  | -0.7267 | 2.3488  | -1.0250 | -0.3284 | 1.0239  | -1.4597 | -1.1290 | 0.6914  | -1.4244 | -1.5344 | 1.0946  | 0.6865  | 0.4048  | -0.4310 |
|1| -1.4469 | -1.6788 | -0.2048 | 1.0333  | -0.7021 | -0.2181 | -0.4096 | 0.1382  | -0.3371 | -0.7821 | -2.2870 | -1.9544 | 0.6715  | 0.5367  | -0.3439 | -0.3719 |
|3| 0.6852  | 0.3316  | 0.7432  | -1.5985 | 1.8693  | -0.6748 | -0.0607 | -0.6371 | -0.7312 | 0.5689  | -1.1195 | 0.0210  | 0.1911  | -0.5772 | -0.4538 | 0.8649  |
|66| 1.8095  | 1.0749  | 0.7773  | -1.0318 | 0.0002  | 0.5312  | 0.4343  | 2.2210  | 0.2996  | 0.8847  | -0.4543 | 1.1797  | -0.0353 | 0.9014  | 1.2481  | -1.0354 |

These numbers are learnable parameters. During training, the model updates these embeddings to capture the relationships between tokens. For example token 1 might frequently appear after token 65 in the dataset and the embedding vectors are updated accordingly to encode this relationship.

### Positional Encoding

Unlike the sequentical models, transformers process the entire sequence in parallel. To allow the model to recognize token order we add positional encodings to token embeddings.

#### Sinusoidal Positional Encoding

For a sequence of length `L` and embedding dimension `d_model`, each position `pos` is encoded as a vector of size `d_model`:

$$
PE_{(pos,2i)} = sin(\frac{pos}{10000^{2i/d_{model}}})
$$

$$
PE_{(pos,2i + 1)} = cos(\frac{pos}{10000^{2i/d_{model}}})
$$

Where:
- `pos` = position in the sequence (0, 1, 2, …)
- `i` = dimension index (0, 1, 2, …, d_model/2 - 1)

Even dimensions use sin, odd dimensions use cos. This produces a deterministic and unique vector for each position, allowing the model to infer relative distances between tokens.


<center>
<img src="./media/pos_encoding.png"></img>
</center>

Before Positional Encoding

| Token ID | Dim 1    | Dim 2    | Dim 3    | Dim 4    | Dim 5    | Dim 6    | Dim 7    | Dim 8    | Dim 9    | Dim 10   | Dim 11   | Dim 12   | Dim 13   | Dim 14   | Dim 15   | Dim 16   |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 65       | -1.0169  | 3.7117   | -3.0031  | 0.0366   | -2.0964  | -4.5249  | -5.8190  | 0.1423   | 6.3528   | -5.9342  | -2.7633  | -6.5703  | -1.8161  | -1.1127  | -3.3698  | 1.6234   |
| 0        | 2.5321   | -0.1290  | -0.9900  | -3.7510  | 4.6948   | 0.8859   | -1.5571  | -2.4139  | -8.1228  | 1.2832   | -0.7425  | 1.6549   | -2.8551  | 2.4007   | 5.5863   | 6.4642   |
| 62       | -1.2332  | -1.0142  | -4.4233  | 1.6503   | 0.4995   | 2.1766   | -4.7730  | -1.4999  | 3.7029   | 0.3568   | 4.4813   | -2.4508  | -3.3141  | 8.0293   | -2.5632  | -2.9621  |
| 65       | -1.0169  | 3.7117   | -3.0031  | 0.0366   | -2.0964  | -4.5249  | -5.8190  | 0.1423   | 6.3528   | -5.9342  | -2.7633  | -6.5703  | -1.8161  | -1.1127  | -3.3698  | 1.6234   |


After Adding Positional Encoding

| Token ID | Dim 1    | Dim 2    | Dim 3    | Dim 4    | Dim 5    | Dim 6    | Dim 7    | Dim 8    | Dim 9    | Dim 10   | Dim 11   | Dim 12   | Dim 13   | Dim 14   | Dim 15   | Dim 16   |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 65       | -1.0169  | 4.7117   | -3.0031  | 1.0366   | -2.0964  | -3.5249  | -5.8190  | 1.1423   | 6.3528   | -4.9342  | -2.7633  | -5.5703  | -1.8161  | -0.1127  | -3.3698  | 2.6234   |
| 0        | 3.3736   | 0.4113   | -0.6790  | -2.8006  | 4.7947   | 1.8809   | -1.5255  | -1.4144  | -8.1128  | 2.2831   | -0.7393  | 2.6549   | -2.8541  | 3.4007   | 5.5866   | 7.4642   |
| 62       | -0.3239  | -1.4303  | -3.8322  | 2.4569   | 0.6982   | 3.1566   | -4.7098  | -0.5019  | 3.7229   | 1.3566   | 4.4877   | -1.4508  | -3.3121  | 9.0293   | -2.5626  | -1.9621  |
| 65       | -0.8758  | 2.7217   | -2.1904  | 0.6194   | -1.8009  | -3.5696  | -5.7243  | 1.1378   | 6.3828   | -4.9347  | -2.7538  | -5.5704  | -1.8131  | -0.1127  | -3.3689  | 2.6234   |

(All of these numbers were taken from the notebook. See `Exercises/main.ipynb`)

Notice that for token `65` and pos = 0. `dim 1` will effected by sin function because `i = 0`.

$$
PE(0,0) = sin(\frac{0}{10000^{2*0/16}}) = 0
$$

So there is no changing.


We have another `65` on pos 3. Let's calculate for `i = 0` again.

$$
PE(3,0) = sin(\frac{3}{10000^{2*0/16}}) = sin(3) = 0.1411
$$

That's why after adding pos encoding, the first dimension of token `65` will be updated to -0.8758.

$$
-1.0169 + 0.1411 = -0.8758
$$

Another way might be to set the positional embeddings as learnable. With this approach, the added values will be learned by the model.

### Resources

- https://www.youtube.com/watch?v=kCc8FmEb1nY& 
- https://www.youtube.com/watch?v=T3OT8kqoqjc&
