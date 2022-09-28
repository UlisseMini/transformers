# Transformers

![gif of training](images/hello-programmer.gif)

Hand-written transformers for learning purposes. Written after reading Karpathy's wonderful [makemore](https://github.com/karpathy/makemore) project. Most of the code is written from memory, with occasional reading of makemore's code when stuck.


## Experiments

### Stacks project

Training on the [stacks project](https://github.com/stacks/stacks-project) (~730k LOC) with ~2M param model (4 layers, 4 heads, 64 dim embeddings.)

Ideally there'd be more data, [Chinchilla](https://arxiv.org/pdf/2203.15556.pdf) found that you want ~10-20x more tokens then parameters for big models, my model is tiny but the direction should be qualitatively correct. Maybe I'll test this.

![loss curves](images/stacks-loss.png)

The trained model can hallucinate some fun stuff. Here's some generated output (each line is an independent generation)

```tex
$\mathcal{O}_{Y, \overline{y}}$ such that $f_{big, *} = f_{small}^{Sh, *}\mathcal{L}$
$\mathcal{O}_X$-module $\mathcal{I}$ we have
Let $Z \to X$ be a morphism of algebraic spaces.
$f_{small}^{-1}\mathcal{F}$ on $f_{big, \etale}\mathcal{O}_Y$.
and this is immediate. We have the same condition as (1) and
\item We say that $|U|$ is a scheme of dimension $\leq 1$ by an open
```

Most of the tex math compiles!

![](images/tex-sample-compiled.png)

I'm gonna add context soon, and see if the model can write a full paper.