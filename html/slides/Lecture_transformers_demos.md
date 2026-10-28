---
theme: default
layout: full
title: "NNDS 2026 — Attention and the transformer: interactive demos"
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

<!-- Written by Claude (2026-10-28): demos for Lecture_transformers.tex, in lecture
     order. GPT-2 data from scripts/make_gpt2_attention.py. -->

<BeamerTitle lecture="Attention and the transformer" subtitle="Interactive demos" />

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import MaskExplorer from '../components/MaskExplorer.vue'
</script>

<BeamerFrame title="Masks and positional biases">

The attention weights of one head. With all scores equal, only the mask and the ALiBi penalty decide where each token looks; with random queries and keys, content and position compete.

<MaskExplorer />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import RopeDial from '../components/RopeDial.vue'
</script>

<BeamerFrame title="Rotary positional embeddings">

RoPE rotates each pair of components of a query and a key by an angle proportional to its position. Move the two positions, or shift both together.

<RopeDial />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import GPT2Heads from '../components/GPT2Heads.vue'
</script>

<BeamerFrame title="The 144 heads of GPT-2">

GPT-2 (small) has 12 layers with 12 heads each. Pick a sentence and a head, then hover a row to see where that token attends.

<GPT2Heads />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import InductionHeads from '../components/InductionHeads.vue'
</script>

<BeamerFrame title="Induction heads">

Fifteen random tokens, repeated twice. In the second copy, an <b>induction head</b> at token $t$ attends to the token after its previous occurrence, and the model copies it ([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)).

<InductionHeads />

</BeamerFrame>

---
layout: full
---

<script setup>
import BeamerFrame from '../components/BeamerFrame.vue'
import RaspPlayground from '../components/RaspPlayground.vue'
</script>

<BeamerFrame title="Thinking like a transformer (RASP)">

In RASP, a small language that maps to transformers, <span class="mono">select</span> builds an attention pattern, <span class="mono">aggregate</span> reads values through it, and elementwise operations are MLPs.

<RaspPlayground />

<div class="research-note">

Weiss et al. (2021), [Thinking like transformers](https://arxiv.org/abs/2106.06981); more programs in the [RASP guide](https://sidn.baulab.info/rasp/) and [interactive notebook](https://srush.github.io/raspy/).

</div>

</BeamerFrame>
