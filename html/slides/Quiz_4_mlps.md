---
theme: default
layout: full
title: "NNDS 2026 — Self-evaluation quiz 4: Fully-connected neural networks"
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

<BeamerTitle lecture="Self-evaluation quiz" subtitle="4. Fully-connected neural networks" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="How this quiz works">

There are 12 questions in total, each graded 1 to 3 stars based on expected difficulty. Questions 10 and 11 are optional: they cover mixtures of experts and looped blocks.

<p>You can retry a question as many times as you like, or ask for the solution. Every solution comes with an explanation.</p>

<div class="definition-box">

The quiz is **not graded**. Your answers are stored only in this browser, and the last slide summarizes them. Question 12 runs Python inside the page, so its first run needs a network connection.

</div>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`Without $\phi$ ($\phi(z)=z$), $f$ is an affine function of $\mathbf{x}$.`, correct: true },
  { text: String.raw`$\phi$ mixes the components of $\mathbf{W}\mathbf{x}+\mathbf{b}$.` },
  { text: String.raw`The output layer is a linear model on the features $\mathbf{h}$.`, correct: true },
  { text: String.raw`$\mathbf{W},\mathbf{b}$ are fixed first, then $\mathbf{w},c$ are fit.` },
  { text: String.raw`Without $\phi$, the network can separate the four XOR points.` },
  { text: String.raw`The number of hidden units is a hyperparameter.`, correct: true },
]
</script>

<BeamerFrame title="1 · The multilayer perceptron  ★">

Consider the MLP $f(\mathbf{x})=\mathbf{w}^\top\mathbf{h}+c$ with $\mathbf{h}=\phi(\mathbf{W}\mathbf{x}+\mathbf{b})$. Which statements are true?

<QuizChoice id="4-q1" :options="options" multiple :columns="2">
<template #explanation>

Without $\phi$ the two affine maps collapse into $(\mathbf{w}^\top\mathbf{W})\mathbf{x}+(\mathbf{w}^\top\mathbf{b}+c)$, which cannot solve XOR. $\phi$ acts elementwise, and all parameters are learned jointly, so the network learns its own features. See [The multilayer perceptron](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=3) and [Some terminology](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=8).

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

An MLP takes inputs with $D_0=784$ features, has two hidden layers of widths $256$ and $128$, and a scalar output $\mathbf{w}^\top\mathbf{h}^{(2)}+c$. How many parameters does it have, biases included?

<QuizInput id="4-q2" kind="number" :expected="233985" :tolerance="0" solution="$233985$">
<template #explanation>

Layer $\ell$ has $D_\ell(D_{\ell-1}+1)$ parameters: $256\cdot 785=200960$ and $128\cdot 257=32896$. The output adds $128+1=129$. The first layer holds most of the parameters, because it sees the full input. See [Adding hidden layers](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=4).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`The penalty adds $\lambda\boldsymbol{\theta}$ to the gradient.`, correct: true },
  { text: String.raw`With $\eta=0.1$, $\lambda=0.01$, each step first multiplies $\boldsymbol{\theta}$ by $0.99$.` },
  { text: String.raw`A parameter whose loss gradient stays zero shrinks towards zero.`, correct: true },
  { text: String.raw`Weight decay removes parameters from the network.` },
  { text: String.raw`In PyTorch, weight decay is an argument of the optimizer.`, correct: true },
]
</script>

<BeamerFrame title="3 · Weight decay  ★">

We train with $\mathcal{L}_\lambda(\boldsymbol{\theta})=\mathcal{L}(\boldsymbol{\theta})+\frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert^2$ and gradient descent with step size $\eta$. Which statements are true?

<QuizChoice id="4-q3" :options="options" multiple :columns="2">
<template #explanation>

One step is $\boldsymbol{\theta}\leftarrow(1-\eta\lambda)\boldsymbol{\theta}-\eta\nabla\mathcal{L}(\boldsymbol{\theta})$, and $1-0.1\cdot 0.01=0.999$. Without a push from the loss, a parameter decays geometrically but never reaches exactly zero. See [Weight decay](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=17).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`One hidden layer can approximate a continuous $g$ on a compact $K$ to any accuracy $\varepsilon$.`, correct: true },
  { text: String.raw`The theorem says how many hidden units are needed for a given $\varepsilon$.` },
  { text: String.raw`It holds for $\phi(s)=s^2$.` },
  { text: String.raw`It holds for ReLU.`, correct: true },
  { text: String.raw`It guarantees that gradient descent finds suitable parameters.` },
  { text: String.raw`It does not imply a small test error from a finite dataset.`, correct: true },
]
</script>

<BeamerFrame title="4 · Universal approximation  ★★">

Which statements about the universal approximation theorem from the lecture are true?

<QuizChoice id="4-q4" :options="options" multiple :columns="2">
<template #explanation>

The theorem needs $\phi$ continuous and non-polynomial: ReLU qualifies, $s^2$ does not. It is an existence result: it says nothing about the width, the data, or whether training finds the parameters. See [Universal approximation](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=7).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`$f(-1,\,1)=0$.` },
  { text: String.raw`Where $x_1>0$ and $x_2>x_1$, $f(\mathbf{x})=x_2$.`, correct: true },
  { text: String.raw`Where $x_1<0$ and $x_2<x_1$, $f(\mathbf{x})=0$.`, correct: true },
  { text: String.raw`$f$ is affine on all of $\mathbb{R}^2$.` },
  { text: String.raw`Training changes the affine maps, but not the regions.` },
  { text: String.raw`Inside a region, the gradient $\nabla f(\mathbf{x})$ is constant.`, correct: true },
]
</script>

<BeamerFrame title="5 · Regions of a ReLU network  ★★">

Consider the network from the lecture, $f(\mathbf{x})=\operatorname{ReLU}(x_1)+\operatorname{ReLU}(x_2-x_1)$. Which statements are true?

<QuizChoice id="4-q5" :options="options" multiple :columns="2">
<template #explanation>

At $(-1,1)$ only the second unit is active: $f=0+2=2$. With both units active, $f=x_1+(x_2-x_1)=x_2$; with neither, $f=0$. The gates are fixed inside a region, so $f$ is affine there. The weights decide both the boundaries and the maps. See [One network, many affine maps](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=12).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="6 · Depth versus width  ★★">

We apply the triangular function $T$ from the lecture five times in a row on $[0,1]$. According to the bound on the slide, at least how many ReLU units does a network with a **single** hidden layer need to compute it?

<QuizInput id="4-q6" kind="number" :expected="31" :tolerance="0" solution="$31$">
<template #explanation>

Five compositions give $2^5=32$ pieces, and $m$ ReLUs in one hidden layer give at most $m+1$ pieces, so $m\geq 31$. The deep network uses $3\cdot 5=15$ units. With ten compositions the numbers are $1023$ and $30$. See [Depth through composition](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=13).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`Equal initial weights keep a layer's hidden units identical.`, correct: true },
  { text: String.raw`ReLU halves the mean square of the preactivations.`, correct: true },
  { text: String.raw`With $\sigma^2=1/D_{\ell-1}$, $m_\ell$ halves at every layer.`, correct: true },
  { text: String.raw`He initialization sets $\sigma^2$ from the output width $D_\ell$.` },
  { text: String.raw`Zero biases break the symmetry between hidden units.` },
  { text: String.raw`With $\sigma^2=4/D_{\ell-1}$, ten layers multiply $m$ by $20$.` },
]
</script>

<BeamerFrame title="7 · Initializing the weights  ★★">

A ReLU MLP has zero biases and weights $W_{ij}\sim\mathcal{N}(0,\sigma^2)$. Let $m_\ell$ be the mean square of the activations at layer $\ell$. Which statements are true?

<QuizChoice id="4-q7" :options="options" multiple :columns="2">
<template #explanation>

Each layer gives $m_\ell=\sigma^2D_{\ell-1}m_{\ell-1}/2$, with $D_{\ell-1}$ its input width. For $\sigma^2=1/D_{\ell-1}$ the factor is $\tfrac12$; for $4/D_{\ell-1}$ it is $2$, so ten layers give $2^{10}=1024$. Random weights break the symmetry. See [Initializing the weights](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=15).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`LeakyReLU with $\alpha=0.01$ maps $s=-2$ to $-0.02$.`, correct: true },
  { text: String.raw`PReLU uses a fixed $\alpha$, chosen before training.` },
  { text: String.raw`$\operatorname{SiLU}(s)$ is never negative.` },
  { text: String.raw`For large positive $s$, $\operatorname{GELU}(s)\approx s$.`, correct: true },
  { text: String.raw`In a GLU, the gate comes from a second projection $\mathbf{W}_g\mathbf{x}$.`, correct: true },
  { text: String.raw`A sigmoid gate can make the multiplier larger than one.` },
]
</script>

<BeamerFrame title="8 · Gates and activations  ★★">

Many activations can be written as $s\,g(s)$ for a gate $g$. Which statements are true?

<QuizChoice id="4-q8" :options="options" multiple :columns="2">
<template #explanation>

LeakyReLU uses $g(s)=\alpha$ for $s<0$; PReLU learns $\alpha$. SiLU is $s\,\sigma(s)$, negative for $s<0$, and $\Phi(s)\rightarrow 1$ makes GELU close to $s$. A sigmoid gate stays in $[0,1]$; SwiGLU's SiLU gate can exceed one. See [ReLU as a gate](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=20) and [GLU and SwiGLU](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=22).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="9 · Parameters of a SwiGLU block  ★★">

A SwiGLU block computes $\mathbf{y}=\mathbf{W}_o\left[(\mathbf{W}_u\mathbf{x})\odot\operatorname{SiLU}(\mathbf{W}_g\mathbf{x})\right]$ with $\mathbf{x}\in\mathbb{R}^{512}$, a hidden size $M=2048$ and no biases. How many parameters does it have?

<QuizInput id="4-q9" kind="number" :expected="3145728" :tolerance="0" solution="$3\cdot 512\cdot 2048=3145728$">
<template #explanation>

$\mathbf{W}_u,\mathbf{W}_g\in\mathbb{R}^{M\times D}$ and $\mathbf{W}_o\in\mathbb{R}^{D\times M}$ give $3DM$. A ReLU block $\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{x})$ with the same $M$ has $2DM=2097152$, so models using SwiGLU often pick a smaller $M$ to keep the count equal. See [GLU and SwiGLU](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=22).

</template>
</QuizInput>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`In a dense MoE, all $8$ experts run on every input.`, correct: true },
  { text: String.raw`In a sparse MoE with $k=2$, each input runs $2$ experts, but all $8$ are stored.`, correct: true },
  { text: String.raw`The router weights $\mathbf{a}(\mathbf{x})$ are the same for every input.` },
  { text: String.raw`In a dense MoE, the router weights sum to $1$.`, correct: true },
  { text: String.raw`With fixed $k$, doubling $E$ doubles the computation per input.` },
  { text: String.raw`The experts can have different output shapes.` },
]
</script>

<BeamerFrame title="10 · Mixtures of experts (optional)  ★★">

A mixture of experts has $E=8$ experts and a router $\mathbf{a}(\mathbf{x})=\operatorname{softmax}(\mathbf{W}_r\mathbf{x})$. Which statements are true?

<QuizChoice id="4-q10" :options="options" multiple :columns="2">
<template #explanation>

The router depends on $\mathbf{x}$, and its softmax weights sum to one. A sparse MoE runs only the top $k$ experts, so its computation grows with $k$ while its memory grows with $E$. The weighted sum needs all experts to have the same output shape. See [Mixtures of experts](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=23).

</template>
</QuizChoice>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const options = [
  { text: String.raw`With separate parameters, the stack has $LP$ parameters.`, correct: true },
  { text: String.raw`With a shared block, the stack has $P$ parameters and needs one evaluation.` },
  { text: String.raw`With a shared block, the stack has $P$ parameters and needs $L$ evaluations.`, correct: true },
  { text: String.raw`The shared block must return the same shape it receives.`, correct: true },
  { text: String.raw`Sharing the parameters makes the forward pass $L$ times faster.` },
]
</script>

<BeamerFrame title="11 · Looping a block (optional)  ★★">

We apply a block with $P$ parameters $L$ times, with separate parameters at each layer or with one shared block. Which statements are true?

<QuizChoice id="4-q11" :options="options" multiple :columns="2">
<template #explanation>

Sharing saves memory, not computation: the block is still evaluated $L$ times. Each output is the next input, so the shapes must match. See [Sharing parameters and looping a block](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=25).

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
  'def mlp(X, hidden, w, c):',
  '    # X: (N, D0); hidden: list of (W, b), W: (D_l, D_{l-1}), b: (D_l,)',
  '    # w: (D_L,), c: scalar; returns the (N,) outputs',
  '    ...',
].join('\n')

const tests = String.raw`
import numpy as np
_rng = np.random.default_rng(0)
def _ref(X, hidden, w, c):
    out = []
    for x in X:
        h = x
        for W, b in hidden:
            h = np.maximum(0.0, W @ h + b)
        out.append(w @ h + c)
    return np.array(out)
for _widths in [[3, 5, 4], [2, 7], [6, 3, 3, 2], [4]]:
    _N = 5
    _X = _rng.normal(size=(_N, _widths[0]))
    _hidden = [(_rng.normal(size=(_widths[i + 1], _widths[i])), _rng.normal(size=_widths[i + 1])) for i in range(len(_widths) - 1)]
    _w = _rng.normal(size=_widths[-1])
    _c = float(_rng.normal())
    _out = mlp(_X, _hidden, _w, _c)
    assert _out is not None, "the function returns None"
    assert np.shape(_out) == (_N,), f"for widths {_widths}: expected shape ({_N},), got {np.shape(_out)}"
    assert np.allclose(_out, _ref(_X, _hidden, _w, _c)), f"for widths {_widths}: the outputs differ from the definition"
`

const solution = [
  'def mlp(X, hidden, w, c):',
  '    for W, b in hidden:',
  '        X = np.maximum(0, X @ W.T + b)',
  '    return X @ w + c',
].join('\n')
</script>

<BeamerFrame title="12 · An MLP forward pass  ★★">

Complete the ReLU MLP below, applied to every row of `X`, with NumPy and no loop over the examples.

<QuizCode id="4-q12" :initial-code="initialCode" :tests="tests" :solution="solution" :code-rows="5">
<template #explanation>

`X @ W.T + b` computes $\mathbf{W}\mathbf{h}+\mathbf{b}$ for every row. See [Adding hidden layers](https://sscardapane.github.io/teaching-materials/pdf/nnds/Lecture_4_Fully_connected.pdf#page=4).

</template>
</QuizCode>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'

const items = [
  { id: '4-q1', title: 'The multilayer perceptron', stars: 1, slide: 3 },
  { id: '4-q2', title: 'Counting parameters', stars: 1, slide: 4 },
  { id: '4-q3', title: 'Weight decay', stars: 1, slide: 5 },
  { id: '4-q4', title: 'Universal approximation', stars: 2, slide: 6 },
  { id: '4-q5', title: 'Regions of a ReLU network', stars: 2, slide: 7 },
  { id: '4-q6', title: 'Depth versus width', stars: 2, slide: 8 },
  { id: '4-q7', title: 'Initializing the weights', stars: 2, slide: 9 },
  { id: '4-q8', title: 'Gates and activations', stars: 2, slide: 10 },
  { id: '4-q9', title: 'Parameters of a SwiGLU block', stars: 2, slide: 11 },
  { id: '4-q10', title: 'Mixtures of experts (optional)', stars: 2, slide: 12 },
  { id: '4-q11', title: 'Looping a block (optional)', stars: 2, slide: 13 },
  { id: '4-q12', title: 'An MLP forward pass', stars: 2, slide: 14 },
]
</script>

<BeamerFrame title="Your answers">

<QuizSummary prefix="4-" :items="items" />

</BeamerFrame>
