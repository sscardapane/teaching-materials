---
theme: default
layout: full
title: "NNDS 2026 — Interactive demos: optimization, convolutions and text"
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

<!-- Written by Claude (2026-10-14): companion demos for 2b and the CNN lecture. -->

<BeamerTitle lecture="Interactive demos" subtitle="Optimization, convolutions and text" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import OptimizerRace from '../components/OptimizerRace.vue'
</script>

<BeamerFrame title="GD, momentum and Adam on a valley">

Raise $\kappa$: GD diverges once $\eta>2/\kappa$, momentum (step $\eta(1-\lambda)$) smooths the zig-zag, and Adam's path barely changes, because it ignores the gradient scale.

<OptimizerRace />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import ConvPlayground from '../components/ConvPlayground.vue'
</script>

<BeamerFrame title="A convolution is a local pattern detector">

Pick a filter or edit its nine weights. The same weights slide over every pixel; with stride 2, we keep one output in four.

<ConvPlayground />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import ReceptiveFieldExplorer from '../components/ReceptiveFieldExplorer.vue'
</script>

<BeamerFrame title="How far can an output see?">

Highlighted units can affect the top output. Compare plain stacking, one stride-2 layer and doubling dilations; make it causal to see what a next-token model may read.

<ReceptiveFieldExplorer />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import BPEStepper from '../components/BPEStepper.vue'
</script>

<BeamerFrame title="Byte pair encoding, one merge at a time">

Each merge joins the most frequent adjacent pair. The vocabulary grows by one token and the corpus gets shorter.

<BPEStepper />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import TemperatureSampler from '../components/TemperatureSampler.vue'
</script>

<BeamerFrame title="Sampling the next token with a temperature">

Small $\tau$ almost always picks the most likely token; large $\tau$ flattens the distribution and the continuations become more varied.

<TemperatureSampler />

</BeamerFrame>
