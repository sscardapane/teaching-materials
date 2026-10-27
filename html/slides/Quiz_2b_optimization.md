---
theme: default
layout: full
title: "NNDS 2026 — Self-evaluation quiz 2b: Derivatives and optimization"
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

<BeamerTitle lecture="Self-evaluation quiz" subtitle="2b. Derivatives and optimization" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="How this quiz works">

There are 11 questions in total, each graded 1 to 3 stars based on expected difficulty. Questions 8 and 10 are on momentum and Adam, the optional part of the lecture.

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

<BeamerFrame title="1 · Jacobian of a composition  ★">

Let $g:\mathbb{R}^{784}\rightarrow\mathbb{R}^{128}$ and $f:\mathbb{R}^{128}\rightarrow\mathbb{R}^{10}$. What is the shape of the Jacobian $\partial\big[f(g(\mathbf{x}))\big]$?

<QuizInput id="2b-q1" kind="shape" :expected="[10, 784]" solution="$(10, 784)$">
<template #explanation>

The Jacobian of a function $\mathbb{R}^D\rightarrow\mathbb{R}^H$ has shape $(H,D)$: one row per output, one column per input. By the chain rule, $\partial f(g(\mathbf{x}))\sim(10,128)$ multiplies $\partial g(\mathbf{x})\sim(128,784)$, and the inner dimensions match. See [Chain rule for Jacobians](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=11).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$(-1,\ 0)$`, correct: true },
  { text: String.raw`$(0,\ 1)$`, correct: true },
  { text: String.raw`$(1,\ 1)$` },
  { text: String.raw`$(1,\ 3)$`, correct: true },
  { text: String.raw`$(-1,\ -3)$` },
  { text: String.raw`$(-2,\ 1)$`, correct: true },
]
</script>

<BeamerFrame title="2 · Descent directions  ★">

At the current parameters $\boldsymbol{\theta}$, the gradient of the loss is $\nabla\mathcal{L}(\boldsymbol{\theta})=(2,\,-1)$. Which of these are descent directions?

<QuizChoice id="2b-q2" :options="options" multiple :columns="3">
<template #explanation>

$\mathbf{p}$ is a descent direction when $\langle\nabla\mathcal{L}(\boldsymbol{\theta}),\mathbf{p}\rangle<0$, that is, when it makes an angle larger than $90°$ with the gradient. The inner products are $-2$, $-1$, $1$, $-1$, $1$ and $-5$. $(1,3)$ descends even though its first coordinate goes against $-\nabla\mathcal{L}$; $(-2,1)=-\nabla\mathcal{L}$ is the steepest one. See [Descent direction](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=19).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`Mini-batch gradients are unbiased.`, correct: true },
  { text: String.raw`Mini-batch gradients equal the full gradient.` },
  { text: String.raw`One update costs the same for any $N$.`, correct: true },
  { text: String.raw`One epoch makes $500$ updates.`, correct: true },
  { text: String.raw`An example can appear twice in one epoch.` },
  { text: String.raw`Averaging the gradients of 4 batches of 100 gives the gradient of their union.`, correct: true },
]
</script>

<BeamerFrame title="3 · Mini-batch SGD  ★">

We train with mini-batch SGD on $N=50{,}000$ examples and batches of $B=100$, shuffling and splitting the dataset at every epoch. Which statements are true?

<QuizChoice id="2b-q3" :options="options" multiple :columns="2">
<template #explanation>

The mini-batch gradient is correct on average but noisy. An update processes $B$ examples, so it costs $O(B)$ whatever $N$ is. Shuffling and splitting uses each example once per epoch, in $N/B=500$ blocks. The gradient is linear, so averaging over equal-sized batches gives the gradient of their union, as with several GPUs. See [Stochastic gradient descent](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=25).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$\mathrm{D}_{\mathbf{u}}f(\mathbf{x})=3$ for $\mathbf{u}=(1,0)$.`, correct: true },
  { text: String.raw`$\partial f(\mathbf{x})/\partial x_1=5$.` },
  { text: String.raw`$\mathrm{D}_{\mathbf{u}}f(\mathbf{x})=5$ for $\mathbf{u}=(0.6,\,0.8)$.`, correct: true },
  { text: String.raw`$\mathrm{D}_{\mathbf{u}}f(\mathbf{x})=-5$ for $\mathbf{u}=(-3,\,-4)$.` },
  { text: String.raw`No unit vector $\mathbf{u}$ gives $\mathrm{D}_{\mathbf{u}}f(\mathbf{x})>5$.`, correct: true },
  { text: String.raw`Moving from $\mathbf{x}$ along $(0.8,\,-0.6)$, $f$ does not change to first order.`, correct: true },
]
</script>

<BeamerFrame title="4 · Directional derivatives  ★★">

At a point $\mathbf{x}$, a function $f:\mathbb{R}^2\rightarrow\mathbb{R}$ has gradient $\nabla f(\mathbf{x})=(3,\,4)$. Which statements are true?

<QuizChoice id="2b-q4" :options="options" multiple :columns="2">
<template #explanation>

All of them follow from $\mathrm{D}_{\mathbf{u}}f(\mathbf{x})=\langle\nabla f(\mathbf{x}),\mathbf{u}\rangle$. Along $\mathbf{e}_1=(1,0)$ we get the partial derivative, $3$. The unit vector $(0.6,0.8)$ points along the gradient and gives the largest value, $\lVert\nabla f(\mathbf{x})\rVert=5$. $(0.8,-0.6)$ is orthogonal to the gradient, along the level set. $(-3,-4)$ is not a unit vector: the definition still applies, but gives $-25$. See [Directional derivative](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=8).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="5 · Predicting a step  ★★">

At the current parameters, $\mathcal{L}(\boldsymbol{\theta})=2$ and $\nabla\mathcal{L}(\boldsymbol{\theta})=(3,\,-4)$. We take one gradient descent step with $\eta=0.01$. Using the first-order approximation, what is the loss after the step?

<QuizInput id="2b-q5" kind="number" :expected="1.75" :tolerance="0.001" solution="$1.75$">
<template #explanation>

The displacement is $\mathbf{h}=-\eta\nabla\mathcal{L}(\boldsymbol{\theta})$, so
$\mathcal{L}(\boldsymbol{\theta}+\mathbf{h})\approx\mathcal{L}(\boldsymbol{\theta})+\langle\nabla\mathcal{L}(\boldsymbol{\theta}),\mathbf{h}\rangle=\mathcal{L}(\boldsymbol{\theta})-\eta\lVert\nabla\mathcal{L}(\boldsymbol{\theta})\rVert^2=2-0.01\cdot 25=1.75$.
The predicted decrease $\eta\lVert\nabla\mathcal{L}\rVert^2$ is positive whenever the gradient is not zero, which is why a small enough step always decreases the loss. See [First-order approximation](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=12).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$|x|$`, correct: true },
  { text: String.raw`$\max(0,\,x)$`, correct: true },
  { text: String.raw`$e^x$`, correct: true },
  { text: String.raw`$x^3$` },
  { text: String.raw`$\sin(x)$` },
  { text: String.raw`$x^4-x^2$` },
]
</script>

<BeamerFrame title="6 · Convex or not?  ★★">

Which of these functions $f:\mathbb{R}\rightarrow\mathbb{R}$ are convex?

<QuizChoice id="2b-q6" :options="options" multiple :columns="3">
<template #explanation>

$|x|$ and $\max(0,x)$ are convex even though their derivative is not defined at $0$: the graph never goes above a chord. $x^3$ and $\sin(x)$ curve downwards on part of the line. $x^4-x^2$ has two minima at $\pm 1/\sqrt{2}$, where it equals $-1/4$: the chord between them stays at $-1/4$, below $f(0)=0$. A convex function cannot have two separate minima with a bump in between. See [Definition of a convex function](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=21).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="7 · Sum instead of mean  ★★">

In a mini-batch training loop with $B=64$, the loss is written as:

<div class="legacy-code">
  <div class="code-line"><span class="ln">1</span><span class="code-text">loss = ((model(xb) - yb) ** 2).sum()     <span class="comment"># instead of .mean()</span></span></div>
  <div class="code-line"><span class="ln">2</span><span class="code-text">loss.backward()</span></div>
</div>

With the same learning rate, how many times larger is each SGD update than with `.mean()`?

<QuizInput id="2b-q7" kind="number" :expected="64" :tolerance="0.001" solution="$64$">
<template #explanation>

The sum is $B$ times the mean, and so is its gradient. Each update is $B$ times larger, as if the learning rate were multiplied by $B$, and changing the batch size now silently changes the learning rate too. The $\frac{1}{B}$ in the mini-batch loss keeps the step size independent of $B$. See [Stochastic gradient descent](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=25).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`With $\lambda=0$, it is gradient descent.`, correct: true },
  { text: String.raw`$\mathbf{u}_t$ depends only on the last two gradients.` },
  { text: String.raw`With a constant gradient $\mathbf{g}$, the steps approach $10\,\eta\lVert\mathbf{g}\rVert$.`, correct: true },
  { text: String.raw`With gradients alternating between $\mathbf{g}$ and $-\mathbf{g}$, the steps are smaller than $\eta\lVert\mathbf{g}\rVert$.`, correct: true },
]
</script>

<BeamerFrame title="8 · Momentum (optional)  ★★">

We run momentum with $\lambda=0.9$ and a constant step size $\eta$. Which statements are true?

<QuizChoice id="2b-q8" :options="options" multiple :columns="2">
<template #explanation>

Unrolling, $\mathbf{u}_t$ sums *all* past gradients with weights $\eta,\eta\lambda,\eta\lambda^2,\dots$ When they agree, the sum tends to $\eta\mathbf{g}/(1-\lambda)$, ten times a GD step. When they alternate, consecutive terms cancel and the steps shrink to $\eta\lVert\mathbf{g}\rVert/(1+\lambda)\approx 0.53\,\eta\lVert\mathbf{g}\rVert$: this is how momentum damps oscillations. See [Momentum](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=31).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$(0,0)$ is a stationary point.` },
  { text: String.raw`$(1,0)$ is a local minimum.`, correct: true },
  { text: String.raw`$(1,0)$ is a global minimum.` },
  { text: String.raw`$(-1,0)$ is a saddle point.`, correct: true },
  { text: String.raw`GD with a small step size, started from $(-1,\,0.5)$, converges to $(-1,0)$.`, correct: true },
  { text: String.raw`GD with a small step size, started from $(-0.99,\,0.5)$, converges to $(-1,0)$.` },
]
</script>

<BeamerFrame title="9 · Stationary points  ★★★">

Consider $\mathcal{L}(\boldsymbol{\theta})=\theta_1^3-3\theta_1+\theta_2^2$. Which statements are true?

<QuizChoice id="2b-q9" :options="options" multiple :columns="2">
<template #explanation>

$\nabla\mathcal{L}=(3\theta_1^2-3,\ 2\theta_2)$ vanishes only at $(\pm1,0)$. Look at each axis separately: $\theta_2^2$ curves upwards, while $\theta_1^3-3\theta_1$ has a local minimum at $1$ and a local maximum at $-1$. So $(1,0)$ is a local minimum, not a global one ($\mathcal{L}\rightarrow-\infty$ as $\theta_1\rightarrow-\infty$), and $(-1,0)$ is a saddle point.

From $(-1,0.5)$ the gradient has no $\theta_1$ component, so GD stops at the saddle; from $\theta_1=-0.99$ it slides to $(1,0)$. This is why gradient noise helps to escape saddles. See [Saddle points in more dimensions](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=17).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`Both coordinates move by about $\eta$.`, correct: true },
  { text: String.raw`The first coordinate moves $10^6$ times more.` },
  { text: String.raw`Without bias correction, the step is the same.` },
  { text: String.raw`Scaling the loss by $100$ leaves the step almost unchanged.`, correct: true },
]
</script>

<BeamerFrame title="10 · Adam's first step (optional)  ★★★">

At the first step of Adam ($t=1$), with default $\beta_1$, $\beta_2$ and a tiny $\varepsilon$, the gradient is $\mathbf{g}_1=(1000,\ 0.001)$. Which statements are true?

<QuizChoice id="2b-q10" :options="options" multiple :columns="2">
<template #explanation>

At $t=1$, the bias correction gives $\widehat{\mathbf{m}}_1=\mathbf{g}_1$ and $\widehat{\mathbf{v}}_1=\mathbf{g}_1\odot\mathbf{g}_1$, so each coordinate moves by $\eta\,|g|/(|g|+\varepsilon)\approx\eta$. Without it, the step would be $(1-\beta_1)/\sqrt{1-\beta_2}\approx 3.16$ times larger. Adam divides the average gradient by its typical size, so scaling the loss changes neither; a GD step would grow 100 times. See [Adam](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=32).

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
  'def grad_mse(w, X, y):',
  '    # X: (N, D), y: (N,), w: (D,)',
  '    # gradient of L(w) = mean((X @ w - y) ** 2), shape (D,)',
  '    ...',
].join('\n')

const tests = String.raw`
import numpy as np
_rng = np.random.default_rng(0)
def _loss(w, X, y):
    return np.mean((X @ w - y) ** 2)
for _N, _D in [(20, 3), (1, 4), (7, 1)]:
    _X = _rng.normal(size=(_N, _D))
    _y = _rng.normal(size=_N)
    _w = _rng.normal(size=_D)
    _g = grad_mse(_w, _X, _y)
    assert _g is not None, "the function returns None"
    _g = np.asarray(_g)
    assert _g.shape == (_D,), f"for N={_N}, D={_D}: expected shape {(_D,)}, got {_g.shape}"
    _h = 1e-6
    _ref = np.array([(_loss(_w + _h * _e, _X, _y) - _loss(_w - _h * _e, _X, _y)) / (2 * _h)
                     for _e in np.eye(_D)])
    assert np.allclose(_g, _ref, rtol=1e-4, atol=1e-6), f"for N={_N}, D={_D}: the values differ from a finite-difference check"
`

const solution = [
  'def grad_mse(w, X, y):',
  '    r = X @ w - y                    # residuals, (N,)',
  '    return 2 * X.T @ r / len(y)      # (D, N) @ (N,) -> (D,)',
].join('\n')
</script>

<BeamerFrame title="11 · The gradient of a loss  ★★★">

Complete the function using NumPy, with no Python loops. It is checked against finite differences on random inputs, including $N=1$ and $D=1$.

<QuizCode id="2b-q11" :initial-code="initialCode" :tests="tests" :solution="solution" :code-rows="6">
<template #explanation>

Write $\mathcal{L}(\mathbf{w})=\frac1N\lVert\mathbf{r}\rVert^2$ with $\mathbf{r}=\mathbf{X}\mathbf{w}-\mathbf{y}$. By the chain rule, the gradient of $\lVert\mathbf{r}\rVert^2$ is $2\mathbf{r}$ and the Jacobian of $\mathbf{r}$ with respect to $\mathbf{w}$ is $\mathbf{X}$, so $\nabla\mathcal{L}(\mathbf{w})=\frac2N\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})$. The tests use the definition of the gradient with a small $h$ along each $\mathbf{e}_i$, a cheap check for any gradient computed by hand. See [Examples of matrix derivatives](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_2_preliminaries.pdf#page=10).

</template>
</QuizCode>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const items = [
  { id: '2b-q1', title: 'Jacobian of a composition', stars: 1, slide: 3 },
  { id: '2b-q2', title: 'Descent directions', stars: 1, slide: 4 },
  { id: '2b-q3', title: 'Mini-batch SGD', stars: 1, slide: 5 },
  { id: '2b-q4', title: 'Directional derivatives', stars: 2, slide: 6 },
  { id: '2b-q5', title: 'Predicting a step', stars: 2, slide: 7 },
  { id: '2b-q6', title: 'Convex or not?', stars: 2, slide: 8 },
  { id: '2b-q7', title: 'Sum instead of mean', stars: 2, slide: 9 },
  { id: '2b-q8', title: 'Momentum (optional)', stars: 2, slide: 10 },
  { id: '2b-q9', title: 'Stationary points', stars: 3, slide: 11 },
  { id: '2b-q10', title: "Adam's first step (optional)", stars: 3, slide: 12 },
  { id: '2b-q11', title: 'The gradient of a loss', stars: 3, slide: 13 },
]
</script>

<BeamerFrame title="Your answers">

<QuizSummary prefix="2b-" :items="items" />

</BeamerFrame>
