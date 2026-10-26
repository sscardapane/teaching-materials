---
theme: default
layout: full
title: "NNDS 2026 — Lecture 2: Optimization in practice"
css: unocss
fonts:
  sans: 'Fira Sans'
  mono: 'Fira Mono'
colorSchema: light
routerMode: hash
themeConfig:
  pageOffset: 29
---

<style>@import '../styles/index.css';</style>

<script setup>
import BeamerTitle from '../components/BeamerTitle.vue'
</script>

<BeamerTitle subtitle="2c. Optimization in practice" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import BlindDescent from '../components/BlindDescent.vue'
</script>

<!-- Written by Claude (2026-10-26): the batch-size and schedule labs were replaced by
     blind descent, the optimizer race and live SGD, at Simone's request. -->

<BeamerFrame title="Optimizing in the dark">

An optimizer never sees the loss: only its value and slope where it stands. Reach the minimum in as few moves as you can, then reveal the curve and compare with GD.

<BlindDescent />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import GradientDescentDemo from '../components/GradientDescentDemo.vue'
</script>

<BeamerFrame title="Gradient descent">

For $f(x)=x^2$, compare the slow, oscillatory, and unstable regimes:

<GradientDescentDemo />

<div class="research-note">

The stability boundary follows directly from $x_{t+1}=(1-2\eta)x_t$: convergence requires $|1-2\eta|<1$.

</div>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import OptimizerRace from '../components/OptimizerRace.vue'
</script>

<BeamerFrame title="GD, momentum and Adam">

Three optimizers, four surfaces. Momentum uses the step $\eta(1-\lambda)$, so it shares GD's long-run step size. More on momentum: Goh (2017), [*Why Momentum Really Works*](https://distill.pub/2017/momentum/).

<OptimizerRace />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import LiveSGD from '../components/LiveSGD.vue'
</script>

<BeamerFrame title="Mini-batch SGD, live">

Logistic regression on 200 points. Each update uses $B$ examples (outlined): compare the noise in the boundary and the loss for different $B$, per update and per example seen.

<LiveSGD />

</BeamerFrame>
