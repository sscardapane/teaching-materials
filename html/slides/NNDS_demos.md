---
theme: default
layout: full
title: "NNDS 2026 — Interactive demos (moved)"
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

<!-- Written by Claude (2026-10-14); retired on 2026-10-26. The demos now live next to
     their lectures, one interactive deck per lecture. This page keeps the old URL working. -->

<BeamerTitle lecture="Interactive demos" subtitle="These demos have moved" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
</script>

<BeamerFrame title="Where the demos are now">

<div class="stack-lg mt-md">
<div class="definition-box">

**Optimization** (gradient descent, GD vs. momentum vs. Adam, mini-batch SGD): [Lecture 2c, *Optimization in practice*](https://sscardapane.github.io/teaching-materials/nnds/Lecture_2_optimization/)

</div>
<div class="definition-box">

**Convolutions and text** (convolution filters, byte pair encoding, the GPT-2 vocabulary, receptive fields): [*Convolutional networks: interactive demos*](https://sscardapane.github.io/teaching-materials/nnds/Lecture_CNNs_demos/)

</div>
</div>

</BeamerFrame>
