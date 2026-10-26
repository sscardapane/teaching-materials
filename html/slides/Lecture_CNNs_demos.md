---
theme: default
layout: full
title: "NNDS 2026 — Convolutional networks: interactive demos"
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

<!-- Written by Claude (2026-10-26): demos linked from Lecture_CNNs.tex, in lecture order.
     Replaces the CNN part of NNDS_demos. -->

<BeamerTitle lecture="Convolutional neural networks" subtitle="Interactive demos" />

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
import BPEStepper from '../components/BPEStepper.vue'
</script>

<BeamerFrame title="Byte pair encoding, one merge at a time">

The corpus of the slides, plus <span class="mono">pun</span> and <span class="mono">bun</span>. The outlined tokens form the most frequent pair, which the next merge joins into one new token.

<BPEStepper />

<div class="research-note">

Next, try a real tokenizer on your own text (numbers, code, Italian): [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/).

</div>

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import VocabExplorer from '../components/VocabExplorer.vue'
</script>

<BeamerFrame title="Inside a trained vocabulary: GPT-2">

Every dot is one of GPT-2's 50,257 tokens, placed by a 2D projection (UMAP) of its trained embedding: similar tokens end up close.

<VocabExplorer />

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
