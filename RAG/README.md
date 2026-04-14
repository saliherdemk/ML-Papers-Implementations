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


## Resources

- https://arxiv.org/pdf/2312.10997
- https://www.youtube.com/watch?v=sVcwVQRHIc8 (I recommend this instead of reading the first resource)
