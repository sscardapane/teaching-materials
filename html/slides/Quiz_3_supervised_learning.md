---
theme: default
layout: full
title: "NNDS 2026 — Self-evaluation quiz 3: Supervised learning and linear models"
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

<BeamerTitle lecture="Self-evaluation quiz" subtitle="3. Supervised learning and linear models" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="How this quiz works">

There are 11 questions in total, each graded 1 to 3 stars based on expected difficulty. Question 10 is optional: it needs some numerical linear algebra and some practice with solving systems of linear equations.

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

const options = [
  { text: String.raw`Hyperparameters are chosen on the validation set.`, correct: true },
  { text: String.raw`The validation set is used to fit $\boldsymbol{\theta}$.` },
  { text: String.raw`Choosing hyperparameters by test loss makes the test loss optimistic.`, correct: true },
  { text: String.raw`The same example can appear in the training and the test set.` },
  { text: String.raw`Moving from one hospital's scans to another's can cause a distribution shift.`, correct: true },
  { text: String.raw`With i.i.d. samples, training and test losses are equal on average.` },
]
</script>

<BeamerFrame title="1 · Training, validation and test  ★">

Which statements are true?

<QuizChoice id="3-q1" :options="options" multiple :columns="2">
<template #explanation>

The training set fits $\boldsymbol{\theta}$, the validation set chooses the hyperparameters, and the disjoint test set is used once at the end. A test set used to make choices becomes part of training, and its loss is biased downwards, like the training loss. See [Training, validation, and test data](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=4).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="2 · Counting parameters  ★">

A linear classifier takes $28\times 28$ grayscale images, flattened into vectors, and has $C=10$ classes. How many parameters does it have, biases included?

<QuizInput id="3-q2" kind="number" :expected="7850" :tolerance="0" solution="$7850$">
<template #explanation>

The inputs have $D=784$ features, so $\mathbf{W}\in\mathbb{R}^{10\times 784}$ and $\mathbf{b}\in\mathbb{R}^{10}$: $7840+10=7850$. Each class has its own row of weights and its own bias. See [Linear models for classification](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=17).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="3 · All logits equal  ★">

A classifier with $C=10$ classes outputs the same value for all logits, for every input. What is its average cross-entropy loss (natural logarithm)?

<QuizInput id="3-q3" kind="number" :expected="2.302585" :tolerance="0.002" solution="$\log 10\approx 2.303$">
<template #explanation>

Equal logits give $p_{ik}=1/10$ for every class, so each example costs $-\log(1/10)=\log 10$, whatever its label. A newly initialized classifier has logits close to each other, so its training loss should start near $\log C$; a much larger value usually signals a bug. See [Multiclass logistic regression](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=20).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`With squared loss, the last example dominates the total.`, correct: true },
  { text: String.raw`Absolute loss gives it a smaller share than squared loss.`, correct: true },
  { text: String.raw`Squared loss is not differentiable at $e=0$.` },
  { text: String.raw`The slope of the absolute loss has the same magnitude at all four errors.`, correct: true },
  { text: String.raw`Absolute loss is more sensitive to the outlier than squared loss.` },
]
</script>

<BeamerFrame title="4 · Losses and outliers  ★★">

A regression model makes errors $e=\hat{y}-y$ equal to $0.1$, $-0.2$, $0.1$ and $10$ on four examples. Which statements are true?

<QuizChoice id="3-q4" :options="options" multiple :columns="2">
<template #explanation>

The last squared error is $100$, against at most $0.04$ for the others, so it dominates the loss and its gradient $2e$. The absolute loss grows linearly with slope $\pm 1$, so every example pulls with the same strength. See [Losses for regression](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=10).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`It has zero empirical risk.`, correct: true },
  { text: String.raw`If $p(x)$ is continuous, it almost always predicts $\bar{y}$ on new inputs.`, correct: true },
  { text: String.raw`The training loss is an unbiased estimate of the expected risk.` },
  { text: String.raw`The loss on unused test data is an unbiased estimate of it.`, correct: true },
  { text: String.raw`A smaller generalization gap means a smaller expected risk.` },
]
</script>

<BeamerFrame title="5 · Empirical and expected risk  ★★">

The lookup table from the lecture returns $y_i$ on training inputs and $\bar{y}$ elsewhere. An estimate is *unbiased* if its average over random datasets equals the true value. Which statements are true?

<QuizChoice id="3-q5" :options="options" multiple :columns="2">
<template #explanation>

A new continuous input matches a training input with probability zero. The training loss is optimistic because $\boldsymbol{\theta}$ was fit on the same examples; unused test data are independent of $\boldsymbol{\theta}$. A small gap with a large training loss still means a large risk. See [Expected risk and empirical risk](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=8).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$p_1=e\cdot p_2$ (Euler's number, $e\approx 2.718$).`, correct: true },
  { text: String.raw`Ignoring finite precision, a small $\tau$ can make a component exactly $0$.` },
  { text: String.raw`Adding $5$ to every logit leaves $\mathbf{p}$ unchanged.`, correct: true },
  { text: String.raw`Doubling every logit leaves $\mathbf{p}$ unchanged.` },
  { text: String.raw`As $\tau\rightarrow\infty$, every component approaches $1/3$.`, correct: true },
  { text: String.raw`Changing $\tau$ can change calibration, but not accuracy.`, correct: true },
]
</script>

<BeamerFrame title="6 · Softmax and temperature  ★★">

Let $\mathbf{p}=\operatorname{softmax}(\mathbf{z}/\tau)$ with $\mathbf{z}=(2,\,1,\,0)$, and $\tau=1$ unless stated otherwise. Which are true?

<QuizChoice id="3-q6" :options="options" multiple :columns="2">
<template #explanation>

Ratios depend only on logit differences, $p_1/p_2=e^{z_1-z_2}=e$, so adding a constant changes nothing. Doubling the logits is the case $\tau=1/2$, which sharpens $\mathbf{p}$. Ignoring finite precision, exponentials are positive; a large $\tau$ flattens $\mathbf{p}$. $\tau$ never changes the ordering of the logits, hence the predicted class. See [Softmax and temperature](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=19) and [Can we trust these probabilities?](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=25).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="7 · A ridge estimate  ★★">

A linear model with one feature and no bias is trained on three examples with inputs $\mathbf{x}=(1,\,2,\,2)$ and targets $\mathbf{y}=(2,\,3,\,5)$. What is the ridge regression solution $w^*$ for $\lambda=3$?

<QuizInput id="3-q7" kind="number" :expected="1.5" :tolerance="0.001" solution="$1.5$">
<template #explanation>

With one feature, $\mathbf{X}^\top\mathbf{X}=1+4+4=9$ and $\mathbf{X}^\top\mathbf{y}=2+6+10=18$, so $w^*=18/(9+3)=1.5$. Least squares ($\lambda=0$) gives $18/9=2$: here the penalty shrinks the weight by a factor $9/12$, and the shrinkage is weaker when the data are larger relative to $\lambda$. See [Regularizing least squares](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=15).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$\mathbf{x}=(0,\,0)$ is assigned to class $1$.`, correct: true },
  { text: String.raw`$\mathbf{x}=(1,\,3)$ has $p=0.5$.`, correct: true },
  { text: String.raw`$\mathbf{x}=(2,\,0)$ is assigned to class $0$.` },
  { text: String.raw`$\mathbf{w}$ is parallel to the decision boundary.` },
  { text: String.raw`Multiplying $\mathbf{w}$ and $b$ by $10$ moves the decision boundary.` },
  { text: String.raw`Multiplying $\mathbf{w}$ and $b$ by $10$ pushes $p$ closer to $0$ or $1$ away from the boundary.`, correct: true },
]
</script>

<BeamerFrame title="8 · A decision boundary  ★★">

A binary logistic regression model has $\mathbf{w}=(2,\,-1)$ and $b=1$, and predicts class $1$ when $p\geq 0.5$. Which statements are true?

<QuizChoice id="3-q8" :options="options" multiple :columns="2">
<template #explanation>

The logits are $s=1$ at $(0,0)$, $s=0$ at $(1,3)$ and $s=5$ at $(2,0)$, so $(2,0)$ is in class $1$ and $(1,3)$ lies on the boundary $2x_1-x_2+1=0$. The boundary is normal to $\mathbf{w}$. Scaling $\mathbf{w}$ and $b$ keeps the zero set of $s$, but makes the sigmoid steeper: the same classifier with more confident probabilities. See [What makes the classifier linear?](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=22).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`Its weights can be fit by least squares.`, correct: true },
  { text: String.raw`Its decision boundary is a straight line.` },
  { text: String.raw`Without $x_1x_2$, some $\mathbf{w}$ separates the four XOR points.` },
  { text: String.raw`With $d=100$ inputs, there are $5050$ degree-two terms.`, correct: true },
  { text: String.raw`Random Fourier features train $\mathbf{a}_j$ and $b_j$ by gradient descent.` },
]
</script>

<BeamerFrame title="9 · Linear in what?  ★★">

Consider $f_{\mathbf{w}}(\mathbf{x})=w_0+w_1x_1+w_2x_2+w_3x_1x_2$ from the XOR example, and more generally $\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x})$. Which statements are true?

<QuizChoice id="3-q9" :options="options" multiple :columns="2">
<template #explanation>

The model is linear in $\mathbf{w}$: least squares works with $\boldsymbol{\phi}(\mathbf{x}_i)$ in place of $\mathbf{x}_i$. For the weights on the slide, the boundary $x_1x_2=0$ is the pair of axes, and no single line separates XOR. Degree two gives $d(d+1)/2$ terms. Random features are sampled once and kept fixed. See [Linear in what?](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=27) and [Fixed nonlinear features](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=28).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`If $N<D$, $\mathbf{X}^\top\mathbf{X}$ is not invertible.`, correct: true },
  { text: String.raw`If $N<D$, the normal equations have no solution.` },
  { text: String.raw`A duplicated feature makes the closed form inapplicable.`, correct: true },
  { text: String.raw`For any $\lambda>0$, $\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}$ is invertible.`, correct: true },
  { text: String.raw`At $\mathbf{w}^*$, $\mathbf{y}-\mathbf{X}\mathbf{w}^*$ is orthogonal to every column of $\mathbf{X}$.`, correct: true },
  { text: String.raw`Ridge ($\lambda>0$) reaches a lower $\mathcal{L}_{\mathrm{LS}}$ than least squares.` },
]
</script>

<BeamerFrame title="10 · Solving least squares (optional)  ★★★">

This question needs some numerical linear algebra. Let $\mathbf{X}\in\mathbb{R}^{N\times D}$ and $\mathbf{y}\in\mathbb{R}^N$, with no bias. Which statements are true?

<QuizChoice id="3-q10" :options="options" multiple :columns="2">
<template #explanation>

$\mathbf{X}^\top\mathbf{X}$ has rank at most $\min(N,D)$, and a duplicated feature loses a rank; the normal equations then have infinitely many solutions. For $\lambda>0$ and $\mathbf{v}\neq\mathbf{0}$, $\mathbf{v}^\top(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})\mathbf{v}=\lVert\mathbf{X}\mathbf{v}\rVert^2+\lambda\lVert\mathbf{v}\rVert^2>0$. The normal equations say $\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w}^*)=\mathbf{0}$. Ridge trades training fit for smaller weights. See [Solving least squares](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=14).

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
  'def cross_entropy(Z, y):',
  '    # Z: (N, C) logits, y: (N,) integer labels in 0, ..., C-1',
  '    ...',
].join('\n')

const tests = String.raw`
import numpy as np
_rng = np.random.default_rng(0)
def _ref(Z, y):
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    return -np.mean(np.log(P[np.arange(len(y)), y]))
for _N, _C in [(8, 3), (1, 5), (6, 2)]:
    _Z = _rng.normal(size=(_N, _C))
    _y = _rng.integers(0, _C, size=_N)
    _out = cross_entropy(_Z, _y)
    assert _out is not None, "the function returns None"
    assert np.ndim(_out) == 0, f"for N={_N}, C={_C}: expected a scalar, got shape {np.shape(_out)}"
    assert np.isclose(_out, _ref(_Z, _y)), f"for N={_N}, C={_C}: the value differs from the definition"
_Z = np.array([[1000.0, 1000.0, 0.0], [-1000.0, 0.0, 0.0]])
_y = np.array([0, 0])
_out = cross_entropy(_Z, _y)
assert np.isfinite(_out), "with logits of size 1000 the result is not finite: the computation overflows"
assert np.isclose(_out, (np.log(2) + 1000 + np.log(2)) / 2), "with logits of size 1000 the value is wrong"
`

const solution = [
  'def cross_entropy(Z, y):',
  '    m = Z.max(axis=1, keepdims=True)',
  '    lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))',
  '    return np.mean(lse - Z[np.arange(len(y)), y])',
].join('\n')
</script>

<BeamerFrame title="11 · Cross-entropy from logits  ★★★">

Complete the function, which returns the mean cross-entropy, using NumPy and no Python loops. It must also work for logits as large as $1000$.

<QuizCode id="3-q11" :initial-code="initialCode" :tests="tests" :solution="solution" :code-rows="5">
<template #explanation>

Each loss is $\operatorname{logsumexp}(\mathbf{z}_i)-z_{i,y_i}$. Subtracting the row maximum keeps the exponentials in $[0,1]$; taking the softmax first gives $e^{1000}=\infty$. See [Cross-entropy from logits](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_3_supervised_learning.pdf#page=24).

</template>
</QuizCode>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const items = [
  { id: '3-q1', title: 'Training, validation and test', stars: 1, slide: 3 },
  { id: '3-q2', title: 'Counting parameters', stars: 1, slide: 4 },
  { id: '3-q3', title: 'All logits equal', stars: 1, slide: 5 },
  { id: '3-q4', title: 'Losses and outliers', stars: 2, slide: 6 },
  { id: '3-q5', title: 'Empirical and expected risk', stars: 2, slide: 7 },
  { id: '3-q6', title: 'Softmax and temperature', stars: 2, slide: 8 },
  { id: '3-q7', title: 'A ridge estimate', stars: 2, slide: 9 },
  { id: '3-q8', title: 'A decision boundary', stars: 2, slide: 10 },
  { id: '3-q9', title: 'Linear in what?', stars: 2, slide: 11 },
  { id: '3-q10', title: 'Solving least squares (optional)', stars: 3, slide: 12 },
  { id: '3-q11', title: 'Cross-entropy from logits', stars: 3, slide: 13 },
]
</script>

<BeamerFrame title="Your answers">

<QuizSummary prefix="3-" :items="items" />

</BeamerFrame>
