# Retrieval-Augmented Generation 

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to produce more accurate, up-to-date and grounded responses. Instead of relying solely on what a model has learned during training, RAG allows the model to look up relevant external information at inference time and use it to generate answers.

$$
\mathrm{Answer}=\mathrm{Generator}(x,\mathrm{Retriever}(x))
$$

Where:

- $x$: user query or input.
- Retriever$(x)$: fetches relevant documents from a knowledge base.
- Generator$(x, \cdot)$: produces the final answer using both the query and retrieved documents.

Instead of memorizing everything, the model learns: how to retrieve useful information and how to use that information effectively.

## Indexing

Prepare documents so they can be retrieved efficiently and relevant pieces can be found quickly at query time. This involves normalizing the format and splitting documents into chunks. In modern RAG, we use two primary strategies.

### Sparse Index (Keyword-based / BM25)

Sparse indexing uses high-dimensional vectors where most values are zero. SOTA systems use BM25, which improves upon simple word counts by balancing term frequency (TF) with inverse document frequency (IDF) and document length.

Pipeline

```
Text → Normalize → Tokenize → Build Inverted Index → BM25 Scoring
```

#### Example

Documents:

| Doc | Text | Length | "cat" freq |
|-----|------|--------|------------|
| Doc 1 | "The cat sat." | 3 | 1 |
| Doc 2 | "A very lazy cat sat on a very warm mat." | 10 | 1 |
| Doc 3 | "The cat and the cat chased the mouse." | 8 | 2 |
| Doc 4 | "Dogs and birds live in the park." | 7 | 0 |

Query: `"cat"`

#### Parameters

- $k_1 = 1.5$ — controls term frequency saturation
- $b = 0.75$ — length normalization penalty
- $\text{avgdl} = \dfrac{3 + 10 + 8 + 7}{4} = 7.0$

#### IDF (Inverse Document Frequency)

It measures how rare a term is across documents. The more documents a term appears in, the lower its score.

Robertson–Spärck Jones formula:

$$
\text{IDF}(t) = \ln\left(\frac{N - df + 0.5}{df + 0.5} + 1\right)
$$

- t: the query term beign scored ("cat")
- N: total number of documents in the corpus (4)
- df: number of documents containing term t (3 - doc 1, 2, 3)
- smoothing constant (0.5): prevents $\ln(0)$ when $df = N$ and keeps IDF positive for very common terms (becuase $df \le N$) 


$$
\text{IDF} 
= \ln\!\left(\frac{4 - 3 + 0.5}{3 + 0.5} + 1\right)
= \ln\!\left(\frac{1.5}{3.5} + 1\right)
= \ln(1.4286)
\approx 0.357
$$

#### Score Formula

$$
\text{Score} = \text{IDF} \times \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \left(1 - b + b \cdot \dfrac{dl}{\text{avgdl}}\right)}
$$

- IDF: how rare the term is across all docs
- $f$: how many times the term appears in this document
- $k_1$: controls how much repeated occurrences of a term can keep boosting the score
- $b$: how aggressively to penalize long documents
- $dl$: length of this specific document
- $\text{avgdl}$: average document length across the corpus

#### Calculations

Doc 1: $dl = 3,\ f = 1$

$$
\text{norm} = 1 - 0.75 + 0.75 \times \frac{3}{7} = 0.571
$$

$$
\text{Score} = 0.357 \times \frac{1 \times 2.5}{1 + 1.5 \times 0.571} = 0.357 \times \frac{2.5}{1.857} = 0.357 \times 1.346 \approx 0.480
$$

Doc 2: $dl = 10,\ f = 1$

$$
\text{norm} = 1 - 0.75 + 0.75 \times \frac{10}{7} = 1.321
$$

$$
\text{Score} = 0.357 \times \frac{1 \times 2.5}{1 + 1.5 \times 1.321} = 0.357 \times \frac{2.5}{2.982} = 0.357 \times 0.838 \approx 0.299
$$

Doc 3: $dl = 8,\ f = 2$

$$
\text{norm} = 1 - 0.75 + 0.75 \times \frac{8}{7} = 1.107
$$

$$
\text{Score} = 0.357 \times \frac{2 \times 2.5}{2 + 1.5 \times 1.107} = 0.357 \times \frac{5}{3.661} = 0.357 \times 1.366 \approx 0.487
$$

Doc 4: $dl = 7,\ f = 0$

$$
\text{norm} = 1 - 0.75 + 0.75 \times \frac{7}{7} = 1.0
$$

$$
\text{Score} = 0.357 \times \frac{0 \times 2.5}{0 + 1.5 \times 1.0} = 0.0
$$

#### Results

| Rank | Doc | Score |
|------|-----|-------|
| 1 | Doc 3 | 0.487 |
| 2 | Doc 1 | 0.480 |
| 3 | Doc 2 | 0.299 |
| 4 | Doc 4 | 0.000 |

Insight: Doc 3 ranks highest because "cat" appears twice and its length is close to $\text{avgdl}$, keeping the length penalty mild. Doc 1 is nearly tied, one mention in a very short document means each token carries more weight. Doc 2 drops because its length (10 tokens) pulls the score down. Doc 4 scores zero, no keyword overlap means no match, regardless of content relevance. This is the core limitation of sparse retrieval.

### Dense Index (Embedding-based / Cosine Similarity)

Dense indexing represents text as low-dimensional vectors where most values are non zero. A neural encoder maps every document and query into the same vector space, so semantically similar texts end up close together even with no keyword overlap.

Pipeline

```
Text → Tokenize → Encoder → Dense Vector → Index → Cosine Similarity
```

#### Example

Same four documents, same query. Let's say encoder procude 3 dimensional vector.

| Doc | Text | Vector |
|-----|------|--------|
| Doc 1 | "The cat sat." | $[0.9,\ 0.2,\ 0.1]$ |
| Doc 2 | "A very lazy cat sat on a very warm mat." | $[0.8,\ 0.3,\ 0.2]$ |
| Doc 3 | "The cat and the cat chased the mouse." | $[0.7,\ 0.5,\ 0.3]$ |
| Doc 4 | "Dogs and birds live in the park." | $[0.1,\ 0.8,\ 0.9]$ |

Query: `"cat"` → $q = [0.85,\ 0.25,\ 0.15]$

#### Score Formula

Cosine similarity measures the angle between two vectors — the smaller the angle, the more similar the meaning.

$$
\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|}
$$

Variables and constants:

- q: query vector produced by the encoder
- d: document vector produced by encoder
- $q \cdot d$: dot product, sum of element-wise products, measures raw alignment
- |q|: L2 norm of the query vector $\sqrt{\sum{q_i^2}}$
- |d|: L2 norm of the query vector $\sqrt{\sum{d_i^2}}$
- $|q| \cdot |d|$: normalization term, divides out vetor magnitude so only direction matters

The result is always in $[-1,\ 1]$. A score of $1$ means identical direction (perfectly similar), $0$ means orthogonal (unrelated), $-1$ means opposite.

Formula directly derived from dot product formula.

$$
a \cdot b = |a| |b| \cos \theta
$$

$$
\cos \theta = \frac{a \cdot b} {|a| |b|}
$$

L2 norm is just length of the vector computed with Pythagoras extended to n dimensions. So when cosine similarity divides by $|q| \cdot |d|$, it's just dividing by the lengths of both vectors to cancel out magnitude. What's left is purely the angle between them, which is what we actually care about for semantic similarity.


#### Calculations

First compute the query norm:

$$
\|q\| = \sqrt{0.85^2 + 0.25^2 + 0.15^2} = \sqrt{0.7225 + 0.0625 + 0.0225} = \sqrt{0.8075} \approx 0.899
$$

Doc 1: $d_1 = [0.9,\ 0.2,\ 0.1]$

$$
q \cdot d_1 = (0.85 \times 0.9) + (0.25 \times 0.2) + (0.15 \times 0.1) = 0.765 + 0.050 + 0.015 = 0.830
$$

$$
\|d_1\| = \sqrt{0.9^2 + 0.2^2 + 0.1^2} = \sqrt{0.81 + 0.04 + 0.01} = \sqrt{0.86} \approx 0.927
$$

$$
\text{sim}(q, d_1) = \frac{0.830}{0.899 \times 0.927} = \frac{0.830}{0.833} \approx 0.996
$$

Doc 2: $d_2 = [0.8,\ 0.3,\ 0.2]$

$$
q \cdot d_2 = (0.85 \times 0.8) + (0.25 \times 0.3) + (0.15 \times 0.2) = 0.680 + 0.075 + 0.030 = 0.785
$$

$$
\|d_2\| = \sqrt{0.8^2 + 0.3^2 + 0.2^2} = \sqrt{0.64 + 0.09 + 0.04} = \sqrt{0.77} \approx 0.877
$$

$$
\text{sim}(q, d_2) = \frac{0.785}{0.899 \times 0.877} = \frac{0.785}{0.788} \approx 0.996
$$

Doc 3: $d_3 = [0.7,\ 0.5,\ 0.3]$

$$
q \cdot d_3 = (0.85 \times 0.7) + (0.25 \times 0.5) + (0.15 \times 0.3) = 0.595 + 0.125 + 0.045 = 0.765
$$

$$
\|d_3\| = \sqrt{0.7^2 + 0.5^2 + 0.3^2} = \sqrt{0.49 + 0.25 + 0.09} = \sqrt{0.83} \approx 0.911
$$

$$
\text{sim}(q, d_3) = \frac{0.765}{0.899 \times 0.911} = \frac{0.765}{0.819} \approx 0.934
$$

Doc 4: $d_4 = [0.1,\ 0.8,\ 0.9]$

$$
q \cdot d_4 = (0.85 \times 0.1) + (0.25 \times 0.8) + (0.15 \times 0.9) = 0.085 + 0.200 + 0.135 = 0.420
$$

$$
\|d_4\| = \sqrt{0.1^2 + 0.8^2 + 0.9^2} = \sqrt{0.01 + 0.64 + 0.81} = \sqrt{1.46} \approx 1.208
$$

$$
\text{sim}(q, d_4) = \frac{0.420}{0.899 \times 1.208} = \frac{0.420}{1.086} \approx 0.387
$$

#### Results

| Rank | Doc | Score |
|------|-----|-------|
| 1 | Doc 1 | 0.996 |
| 2 | Doc 2 | 0.996 |
| 3 | Doc 3 | 0.934 |
| 4 | Doc 4 | 0.387 |

Insight: Unlike BM25, Doc 4 still receives a non-zero score (0.387) even though it contains no mention of "cat" because the encoder captures semantic proximity between animal related concepts. The scores are driven entirely by vector direction and not keyword overlap.

## Retrieval
 
Once documents are indexed and scored, retrieval is straightforward: run the query through BM25 or an encoder, score all documents, sort, and return the top $k$ results.
 
### Hybrid Retrieval and Reciprocal Rank Fusion (RRF)
 
BM25 and dense retrieval have complementary strengths. BM25 is precise on exact keyword matches, dense handles semantic similarity. Instead of choosing one, hybrid retrieval runs both and merges the ranked lists using Reciprocal Rank Fusion. (https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
 
RRF doesn't use scores at all. It only looks at rank positions, which makes it robust to the fact that BM25 and cosine similarity scores are on completely different scales.
 
$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$
 
- $d$: the document being scored
- $R$: the set of ranked lists (e.g. one from BM25, one from dense)
- $\text{rank}_r(d)$: position of document $d$ in ranked list $r$ (1-indexed)
- $k$: smoothing constant, typically 60, prevents top ranked documents from dominating too heavily

Each document accumulates a score across all lists. A document ranked 1st in both lists scores higher than one that only appears in one. Final results are sorted by RRF score descending and the top $k$ are passed to the generator.

### Example
 
Reusing the ranked results from BM25 and dense retrieval above:
 
| Doc | BM25 Rank | Dense Rank |
|-----|-----------|------------|
| Doc 1 | 2 | 1 |
| Doc 2 | 3 | 2 |
| Doc 3 | 1 | 3 |
| Doc 4 | 4 | 4 |
 
Using $k = 60$:
 
$$
\text{RRF}(d) = \frac{1}{60 + \text{rank}_\text{BM25}(d)} + \frac{1}{60 + \text{rank}_\text{dense}(d)}
$$
 
Doc 1: $\dfrac{1}{62} + \dfrac{1}{61} = 0.01613 + 0.01639 \approx 0.03252$
 
Doc 2: $\dfrac{1}{63} + \dfrac{1}{62} = 0.01587 + 0.01613 \approx 0.03200$
 
Doc 3: $\dfrac{1}{61} + \dfrac{1}{63} = 0.01639 + 0.01587 \approx 0.03226$
 
Doc 4: $\dfrac{1}{64} + \dfrac{1}{64} = 0.01563 + 0.01563 \approx 0.03126$
 
### Results
 
| Rank | Doc | RRF Score |
|------|-----|-----------|
| 1 | Doc 1 | 0.03252 |
| 2 | Doc 3 | 0.03226 |
| 3 | Doc 2 | 0.03200 |
| 4 | Doc 4 | 0.03126 |

## Generation
 
Once the top $k$ documents are retrieved, they are injected into a prompt alongside the user's query and passed to the LLM. The model then generates an answer grounded in the retrieved context rather than relying solely on what it learned during training.
 
 
```
You are a helpful assistant. Use the context below to answer the question.
 
Context:
[Doc 1]: ...
[Doc 2]: ...
[Doc 3]: ...
 
Question: {user query}
 
Answer:
```
 
The retrieved documents fill the context block in ranked order, highest RRF score first. The model reads the context and the question together and generates a response.
 
## Query Translation

 Some techniques emerged to retrieve the related documents better.

### Multi Query

You basically ask llm to generate multiple queries based on the user query.

Query: "Antibiotic resistance patterns in community acquired pneumonia"

LLM:

- What are the common antibiotic resistance trends observed in community-acquired pneumonia?
- How does antibiotic resistance manifest in cases of pneumonia acquired outside of healthcare settings?
- What are the current patterns of antibiotic resistance for community-acquired pneumonia?
- Describe the prevalence of antibiotic resistance in community-acquired pneumonia.
- What are the typical resistance profiles of bacteria causing community-acquired pneumonia?

Then we retrieve documents for each of the query and then deduplicate the documents.

### Rag-Fusion

RAG Fusion extends Multi Query by replacing deduplication with RRF. Instead of just pooling unique documents, it retrieves a ranked list for each sub-query and merges those lists using RRF so that documents consistently appearing near the top across multiple queries rise to the front.

Query: "Antibiotic resistance patterns in community acquired pneumonia"

LLM generates the same sub-queries as Multi Query. Then for each sub-query we run retrieval and get a ranked list:

| Doc | Rank (Q1) | Rank (Q2) | Rank (Q3) | Rank (Q4) | Rank (Q5) |
|-----|-----------|-----------|-----------|-----------|-----------|
| Doc A | 1 | 2 | 1 | 3 | 2 |
| Doc B | 2 | 1 | 4 | 1 | 3 |
| Doc C | 3 | 3 | 2 | 2 | 1 |
| Doc D | 4 | 5 | 3 | 4 | 5 |
| Doc E | 5 | 4 | 5 | 5 | 4 |

Using $k = 60$, each document's RRF score sums across all five ranked lists:

$$
\text{RRF}(d) = \sum_{i=1}^{5} \frac{1}{60 + \text{rank}_i(d)}
$$

Doc A: $\dfrac{1}{61} + \dfrac{1}{62} + \dfrac{1}{61} + \dfrac{1}{63} + \dfrac{1}{62} \approx 0.08138$

Doc B: $\dfrac{1}{62} + \dfrac{1}{61} + \dfrac{1}{64} + \dfrac{1}{61} + \dfrac{1}{63} \approx 0.08091$

Doc C: $\dfrac{1}{63} + \dfrac{1}{63} + \dfrac{1}{62} + \dfrac{1}{62} + \dfrac{1}{61} \approx 0.08020$

Doc D: $\dfrac{1}{64} + \dfrac{1}{65} + \dfrac{1}{63} + \dfrac{1}{64} + \dfrac{1}{65} \approx 0.07732$

Doc E: $\dfrac{1}{65} + \dfrac{1}{64} + \dfrac{1}{65} + \dfrac{1}{65} + \dfrac{1}{64} \approx 0.07660$

#### Results

| Rank | Doc | RRF Score |
|------|-----|-----------|
| 1 | Doc A | 0.08138 |
| 2 | Doc B | 0.08091 |
| 3 | Doc C | 0.08020 |
| 4 | Doc D | 0.07732 |
| 5 | Doc E | 0.07660 |

The top $k$ documents by RRF score are then passed to the generator. A document that ranks 1st for one query but never appears for the others will score lower than one that ranks 2nd or 3rd consistently across all queries.
## Resources

- https://arxiv.org/pdf/2312.10997
- https://www.youtube.com/watch?v=sVcwVQRHIc8 (I recommend this instead of reading the first resource)
- https://www.youtube.com/watch?v=0iNrGpwZwog
