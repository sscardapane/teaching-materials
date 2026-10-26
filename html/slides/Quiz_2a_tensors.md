---
theme: default
layout: full
title: "NNDS 2026 — Self-evaluation quiz 2a: Tensors and linear maps"
css: unocss
fonts:
  sans: 'Fira Sans'
  mono: 'Fira Mono'
colorSchema: light
routerMode: hash
---

<style>@import '../styles/index.css';</style>

<script setup>
import BeamerTitle from '../components/BeamerTitle.vue'
</script>

<BeamerTitle lecture="Self-evaluation quiz" subtitle="2a. Tensors and linear maps" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="How this quiz works">

There are 11 questions in total, each graded 1 to 3 stars based on expected difficulty.

<p>You can retry a question as many times as you like, or ask for the solution. Every solution comes with an explanation.</p>

<div class="definition-box">

The quiz is **not graded**. Your answers are stored only in this browser, and the last slide summarizes them. Question 11 runs Python inside the page, so its first run needs a network connection.

</div>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="1 · Slicing and reductions  ★">

Consider a batch of sequences $X\sim(B,T,D)$:

<div class="legacy-code">
  <div class="code-line"><span class="ln">1</span><span class="code-text">X = torch.randn(32, 20, 128)            <span class="comment"># (B, T, D)</span></span></div>
  <div class="code-line"><span class="ln">2</span><span class="code-text">Y = X[:, 3:7, ::2].mean(dim=1)</span></div>
</div>

What is the shape of `Y`?

<QuizInput id="2a-q1" kind="shape" :expected="[32, 64]" solution="$(32, 64)$">
<template #explanation>

A slice keeps its axis: `3:7` selects 4 tokens and `::2` every other feature, so `X[:, 3:7, ::2]` has shape $(32, 4, 64)$. The mean over `dim=1` then removes the token axis. Compare with an integer index: `X[:, 3]` drops the axis, while `X[:, 3:4]` keeps it with size 1. See [Indexing tensors](https://sscardapane.github.io/teaching-materials/nnds/Lecture_2_tensors/#/5).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: '<code>(4,)</code>', correct: true },
  { text: '<code>(2, 4)</code>' },
  { text: '<code>(3, 1)</code>', correct: true },
  { text: '<code>(4, 3)</code>' },
  { text: '<code>(2, 1, 1)</code>', correct: true },
  { text: '<code>(5, 1, 1, 1)</code>', correct: true },
]
</script>

<BeamerFrame title="2 · Broadcasting  ★">

`X` has shape `(2, 3, 4)`. For which shapes of `Y` does `X + Y` run without an error?

<QuizChoice id="2a-q2" :options="options" multiple :columns="3">
<template #explanation>

Right-align the shapes and compare sizes pairwise: each pair must be equal or contain a 1.

- `(2, 4)` aligns 2 with 3, and `(4, 3)` aligns 4 with 3 and 3 with 4: both fail.
- `(5, 1, 1, 1)` works, and the output has shape `(5, 2, 3, 4)`. Broadcasting can also *add* axes to the result, which is a frequent source of silently wrong shapes (see the next questions).

See [Broadcasting rules](https://sscardapane.github.io/teaching-materials/nnds/Lecture_2_tensors/#/11).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="3 · Masked mean pooling  ★★">

A batch of 4 sentences, padded to 10 tokens with ID 0, and an embedding table for 5000 tokens:

<div class="legacy-code">
  <div class="code-line"><span class="ln">1</span><span class="code-text">token_ids = torch.randint(5000, (4, 10))</span></div>
  <div class="code-line"><span class="ln">2</span><span class="code-text">E = torch.randn(5000, 64)</span></div>
  <div class="code-line"><span class="ln">3</span><span class="code-text">emb = E[token_ids]</span></div>
  <div class="code-line"><span class="ln">4</span><span class="code-text">mask = token_ids != 0</span></div>
  <div class="code-line"><span class="ln">5</span><span class="code-text">total = (emb * mask[..., None]).sum(dim=1)</span></div>
  <div class="code-line"><span class="ln">6</span><span class="code-text">pooled = total / mask.sum(dim=1, keepdim=True)</span></div>
</div>

What is the shape of `pooled`?

<QuizInput id="2a-q3" kind="shape" :expected="[4, 64]" solution="$(4, 64)$">
<template #explanation>

Indexing `E` with an integer tensor of shape $(4, 10)$ gives one row of `E` per token: `emb` has shape $(4, 10, 64)$. `mask[..., None]` has shape $(4, 10, 1)$ and broadcasts over the features, zeroing the padding. Summing over the tokens gives `total` with shape $(4, 64)$, and `mask.sum(dim=1, keepdim=True)` has shape $(4, 1)$, which broadcasts again. The result is the average embedding of the *non-padding* tokens of each sentence.

Without `keepdim=True`, the division would be $(4, 64) / (4)$: right-aligning 64 with 4 raises an error.

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: 'It computes the mean squared error, as intended.' },
  { text: 'It raises a shape error.' },
  { text: String.raw`It runs, but averages $(\hat y_i - y_j)^2$ over all $B^2$ pairs $(i, j)$.`, correct: true },
  { text: String.raw`It runs, but computes the mean squared error divided by $B$.` },
]
</script>

<BeamerFrame title="4 · A silent bug  ★★">

A regression model returns one prediction per example:

<div class="legacy-code">
  <div class="code-line"><span class="ln">1</span><span class="code-text">y_pred = model(X)                  <span class="comment"># shape (B, 1)</span></span></div>
  <div class="code-line"><span class="ln">2</span><span class="code-text">y = targets                        <span class="comment"># shape (B,)</span></span></div>
  <div class="code-line"><span class="ln">3</span><span class="code-text">loss = ((y_pred - y) ** 2).mean()</span></div>
</div>

What does line 3 compute?

<QuizChoice id="2a-q4" :options="options" :columns="2">
<template #explanation>

Right-aligned, the shapes are $(B, 1)$ and $(B)$: the pairs are $(1, B)$ and $(B,\ \text{missing})$, so the result has shape $(B, B)$ with entry $(i, j)$ equal to $\hat y_i - y_j$. No error is raised. The minimizer of this loss predicts the *mean target* for every input, so training "works" but learns nothing useful.

Fix it by matching the shapes explicitly: `y_pred.squeeze(1) - y`, or `y_pred - y[:, None]`.

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: '<code>X @ W + b</code>' },
  { text: '<code>torch.softmax(X, dim=1)</code>' },
  { text: '<code>torch.softmax(X, dim=0)</code>', correct: true },
  { text: '<code>X - X.mean(dim=0)</code>', correct: true },
  { text: '<code>X / X.norm(dim=1, keepdim=True)</code>' },
  { text: '<code>X.cumsum(dim=0)</code>', correct: true },
]
</script>

<BeamerFrame title="5 · Independence across the batch  ★★">

Let $\mathbf{X}\sim(B,D)$ be a batch, $\mathbf{W}\sim(D,H)$ and $\mathbf{b}\sim(H)$. In which operations does row $b$ of the output depend on rows of `X` **other than** $\mathbf{x}_b$?

<QuizChoice id="2a-q5" :options="options" multiple :columns="2">
<template #explanation>

An operation mixes examples exactly when it reduces or scans along the batch axis (`dim=0`): the softmax over `dim=0`, the batch mean and the cumulative sum. Then the prediction for one input changes with the other inputs that happen to share its batch. The other three act on each row separately. See [Matrix multiplication](https://sscardapane.github.io/teaching-materials/nnds/Lecture_2_tensors/#/8).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$\operatorname{tr}(\mathbf{X}^\top\mathbf{Y})$`, correct: true },
  { text: String.raw`$\operatorname{tr}(\mathbf{X}\mathbf{Y})$` },
  { text: String.raw`$\mathbf{1}_m^\top(\mathbf{X}\odot\mathbf{Y})\,\mathbf{1}_n$`, correct: true },
  { text: String.raw`$\mathbf{1}_m^\top\mathbf{X}\mathbf{Y}^\top\mathbf{1}_m$` },
  { text: '<code>einsum("ij,ij->", X, Y)</code>', correct: true },
  { text: '<code>einsum("ij,kj->", X, Y)</code>' },
]
</script>

<BeamerFrame title="6 · The matrix inner product  ★★">

Let $\mathbf{X},\mathbf{Y}\sim(m,n)$ with $m\neq n$, and let $\mathbf{1}_k$ be the vector of $k$ ones. Which expressions are equal to $\sum_i\sum_j X_{i,j}Y_{i,j}$?

<QuizChoice id="2a-q6" :options="options" multiple :columns="2">
<template #explanation>

$\operatorname{tr}(\mathbf{X}^\top\mathbf{Y})=\sum_j[\mathbf{X}^\top\mathbf{Y}]_{j,j}=\sum_j\sum_i X_{i,j}Y_{i,j}$, and summing all entries of the Hadamard product $\mathbf{X}\odot\mathbf{Y}$ gives the same thing, as does `"ij,ij->"`. $\mathbf{X}\mathbf{Y}$ is not defined when $m\neq n$. Finally, $\mathbf{1}^\top\mathbf{X}\mathbf{Y}^\top\mathbf{1}=\sum_{i,k}\sum_j X_{i,j}Y_{k,j}$ pairs every row of $\mathbf{X}$ with *every* row of $\mathbf{Y}$, which is what `"ij,kj->"` computes too. See [More array operations](https://sscardapane.github.io/teaching-materials/nnds/Lecture_2_tensors/#/9).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="7 · Writing an einsum  ★★">

Given batches of matrices $X\sim(B,T,D)$ and $Y\sim(B,S,D)$, compute all inner products between rows of $X$ and rows of $Y$ within each batch element:

$$
A_{b,t,s}=\langle [X]_{b,t,:},\,[Y]_{b,s,:}\rangle,\qquad A\sim(B,T,S).
$$

Write the `einsum` specification, so that `A = torch.einsum(spec, X, Y)`.

<QuizInput id="2a-q7" kind="einsum" expected="btd,bsd->bts" solution="<code>btd,bsd->bts</code> (any index names)">
<template #explanation>

Indices that appear in the output are kept ($b$, $t$, $s$); indices that appear only in the inputs are summed ($d$). Since $b$ appears in both inputs *and* in the output, it acts as a batch axis rather than being summed. This is the same as the batched product `X @ Y.transpose(1, 2)`, and it is the first step of the attention layers we will see later in the course.

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: 'It raises an error, because <code>t</code> is missing from the output.' },
  { text: '<code>(X @ W)[:, -1]</code>, the last token only' },
  { text: '<code>X.mean(dim=1) @ W</code>' },
  { text: '<code>X.sum(dim=1) @ W</code>', correct: true },
]
</script>

<BeamerFrame title="8 · A missing index  ★★★">

For $X\sim(B,T,D)$ and $\mathbf{W}\sim(D,H)$, consider:

<div class="legacy-code">
  <div class="code-line"><span class="ln">1</span><span class="code-text">Y = torch.einsum(<span class="comment">"btd,dh-&gt;bh"</span>, X, W)</span></div>
</div>

Which expression gives the same `Y`?

<QuizChoice id="2a-q8" :options="options" :columns="2">
<template #explanation>

An index missing from the output is summed, so $Y_{b,h}=\sum_t\sum_d X_{b,t,d}W_{d,h}$. Exchanging the two sums, $Y_{b,h}=\sum_d\big(\sum_t X_{b,t,d}\big)W_{d,h}$: summing the tokens and then applying $\mathbf{W}$ is the same as applying $\mathbf{W}$ to each token and summing the results, because the map is linear. See [Tensor contractions](https://sscardapane.github.io/teaching-materials/nnds/Lecture_2_tensors/#/10).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="9 · The order of products  ★★★">

Let $\mathbf{x}\sim(D)$ and $\mathbf{W}_1,\mathbf{W}_2\sim(D,D)$ with $D=1000$. Multiplying an $(a,b)$ matrix by a $(b,c)$ matrix takes $abc$ scalar multiplications. Both of these give the same vector:

$$
\mathbf{x}^\top(\mathbf{W}_1\mathbf{W}_2)
\qquad\text{and}\qquad
(\mathbf{x}^\top\mathbf{W}_1)\,\mathbf{W}_2 .
$$

How many times more multiplications does the left one need?

<QuizInput id="2a-q9" kind="number" :expected="500.5" :tolerance="0.02" solution="$(D+1)/2 = 500.5$, so about 500 times">
<template #explanation>

On the left, $\mathbf{W}_1\mathbf{W}_2$ costs $D^3$ and the product with $\mathbf{x}^\top$ another $D^2$. On the right, both steps are vector-matrix products of cost $D^2$. The ratio is $(D^3+D^2)/(2D^2)=(D+1)/2$. Matrix multiplication is associative, but its cost is not: the same observation explains why gradients are computed in reverse order (backpropagation), as we will see with automatic differentiation.

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const W = [[0, 0.5], [-1, 0]]

const options = [
  { text: String.raw`$\begin{bmatrix}0 & -1\\ 0.5 & 0\end{bmatrix}$` },
  { text: String.raw`$\begin{bmatrix}0 & 0.5\\ -1 & 0\end{bmatrix}$`, correct: true },
  { text: String.raw`$\begin{bmatrix}0 & 1\\ -0.5 & 0\end{bmatrix}$` },
  { text: String.raw`$\begin{bmatrix}0 & -0.5\\ 1 & 0\end{bmatrix}$` },
]
</script>

<BeamerFrame title="10 · Reading a linear map  ★★★">

Each point of the shape on the left is a row of $\mathbf{X}\sim(N,2)$; the right shows the rows of $\mathbf{X}\mathbf{W}$, as in our layers. Which $\mathbf{W}\sim(2,2)$ was used?

<QuizChoice id="2a-q10" :options="options" :columns="2">
<template #aside>
  <LinearMapFigure :matrix="W" />
</template>
<template #explanation>

With row vectors, $\mathbf{x}^\top\mathbf{W}$ combines the **rows** of $\mathbf{W}$: the image of $(1,0)$ is the first row, and the image of $(0,1)$ the second. The red point $(1.2, 2)$ ends at $1.2\cdot(0, 0.5)+2\cdot(-1, 0)=(-2, 0.6)$. The first option is the transpose: it would draw this picture with column vectors ($\mathbf{W}\mathbf{x}$), but with row vectors it sends the red point to $(1, -1.2)$.

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const initialCode = [
  'import numpy as np',
  '',
  'def pairwise_sq_dists(X, Y):',
  '    # X: (N, D), Y: (M, D)  ->  (N, M), with',
  '    # out[i, j] = ||X[i] - Y[j]||^2, without Python loops',
  '    ...',
].join('\n')

const tests = String.raw`
import numpy as np
_rng = np.random.default_rng(0)
for _N, _M, _D in [(5, 7, 3), (1, 4, 2), (6, 1, 8)]:
    _X = _rng.normal(size=(_N, _D))
    _Y = _rng.normal(size=(_M, _D))
    _out = pairwise_sq_dists(_X, _Y)
    assert _out is not None, "the function returns None"
    _out = np.asarray(_out)
    assert _out.shape == (_N, _M), f"for N={_N}, M={_M}: expected shape {(_N, _M)}, got {_out.shape}"
    _ref = np.array([[((x - y) ** 2).sum() for y in _Y] for x in _X])
    assert np.allclose(_out, _ref), f"for N={_N}, M={_M}: the values differ from the definition"
`

const solution = [
  'def pairwise_sq_dists(X, Y):',
  '    # broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D)',
  '    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)',
  '',
  '# lighter on memory: ||x||^2 + ||y||^2 - 2<x, y>',
  '#   (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T',
].join('\n')
</script>

<BeamerFrame title="11 · Pairwise distances  ★★★">

Complete the function using NumPy, with no Python loops. It is tested on random inputs, including $N=1$ and $M=1$.

<QuizCode id="2a-q11" :initial-code="initialCode" :tests="tests" :solution="solution" :code-rows="6">
<template #explanation>

Inserting axes of size 1 lets broadcasting form all $N\cdot M$ differences at once, but it builds an $(N,M,D)$ intermediate. Expanding $\lVert\mathbf{x}-\mathbf{y}\rVert^2=\lVert\mathbf{x}\rVert^2+\lVert\mathbf{y}\rVert^2-2\langle\mathbf{x},\mathbf{y}\rangle$ needs only $(N,M)$ memory and one matrix product, which is how libraries compute it (in floating point it can come out slightly negative, so it is often clipped at zero).

</template>
</QuizCode>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const items = [
  { id: '2a-q1', title: 'Slicing and reductions', stars: 1, slide: 3 },
  { id: '2a-q2', title: 'Broadcasting', stars: 1, slide: 4 },
  { id: '2a-q3', title: 'Masked mean pooling', stars: 2, slide: 5 },
  { id: '2a-q4', title: 'A silent bug', stars: 2, slide: 6 },
  { id: '2a-q5', title: 'Independence across the batch', stars: 2, slide: 7 },
  { id: '2a-q6', title: 'The matrix inner product', stars: 2, slide: 8 },
  { id: '2a-q7', title: 'Writing an einsum', stars: 2, slide: 9 },
  { id: '2a-q8', title: 'A missing index', stars: 3, slide: 10 },
  { id: '2a-q9', title: 'The order of products', stars: 3, slide: 11 },
  { id: '2a-q10', title: 'Reading a linear map', stars: 3, slide: 12 },
  { id: '2a-q11', title: 'Pairwise distances', stars: 3, slide: 13 },
]
</script>

<BeamerFrame title="Your answers">

<QuizSummary prefix="2a-" :items="items" />

</BeamerFrame>
