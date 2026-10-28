<!-- Written by Claude (2026-10-28): induction heads in GPT-2 (small), for the
     Transformer demos. Fifteen random tokens repeated twice: the loss drops on
     the second copy, and the induction heads attend to the token right after
     the previous occurrence of the current token. Data from
     scripts/make_gpt2_attention.py (html/data/gpt2_attention.json). -->
<template>
  <div class="ih" v-if="rep">
    <div class="ih-left">
      <div class="ih-heads">
        <button v-for="p in presets" :key="p.l + '.' + p.h" :class="{ on: sel[0] === p.l && sel[1] === p.h }"
          @click="sel = [p.l, p.h]">layer {{ p.l }}, head {{ p.h }} <span class="muted">{{ p.name }}</span></button>
      </div>
      <svg :width="x0 + T * c + 4" :height="y0 + T * c + 4" role="img" aria-label="Attention on the repeated sequence">
        <text :x="x0 + (L / 2 + 1) * c" :y="y0 - 6" text-anchor="middle" class="ax">first copy</text>
        <text :x="x0 + (1.5 * L + 1) * c" :y="y0 - 6" text-anchor="middle" class="ax">second copy</text>
        <text :x="x0 - 6" :y="y0 + (L / 2 + 1) * c" text-anchor="end" class="ax">first</text>
        <text :x="x0 - 6" :y="y0 + (1.5 * L + 1) * c" text-anchor="end" class="ax">second</text>
        <g v-for="i in T" :key="i" @mouseenter="hover = i - 1">
          <rect v-for="j in i" :key="j" :x="x0 + (j - 1) * c" :y="y0 + (i - 1) * c" :width="c" :height="c"
            :fill="shade(w(i - 1, j - 1))" />
        </g>
        <line :x1="x0 + (L + 1) * c" :y1="y0" :x2="x0 + (L + 1) * c" :y2="y0 + T * c" stroke="#d79b00" stroke-dasharray="3 3" />
        <line :x1="x0" :y1="y0 + (L + 1) * c" :x2="x0 + T * c" :y2="y0 + (L + 1) * c" stroke="#d79b00" stroke-dasharray="3 3" />
        <rect v-if="hover >= 0" :x="x0" :y="y0 + hover * c" :width="(hover + 1) * c" :height="c" fill="none" stroke="#8c0000" stroke-width="1.5" />
      </svg>
    </div>

    <div class="ih-right">
      <div class="ih-seq">
        <span v-for="(t, j) in rep.tokens" :key="j" class="wt" :class="{ q: j === hover, half: j === L + 1 }"
          :style="{ background: j <= hover ? shade(w(hover, j)) : 'transparent', color: j <= hover && w(hover, j) > 50 ? '#fff' : '#222' }">{{ vis(t) }}</span>
      </div>
      <div class="muted">Hover a row of the matrix to change the query (outlined).</div>

      <svg :width="BW" :height="BH" role="img" aria-label="Loss per position">
        <rect v-for="(v, i) in rep.loss" :key="i" :x="bx(i)" :y="by(v)" :width="bw - 2" :height="BH - 22 - by(v)"
          :style="{ fill: i < L ? '#9ba8aa' : '#08457d' }" />
        <line x1="0" :y1="BH - 22" :x2="BW" :y2="BH - 22" stroke="#9ba8aa" />
        <text :x="bx(L / 2)" :y="BH - 6" text-anchor="middle" class="ax">first copy: loss {{ mean(0, L).toFixed(1) }}</text>
        <text :x="bx(1.5 * L)" :y="BH - 6" text-anchor="middle" class="ax">second copy: loss {{ mean(L, 2 * L).toFixed(1) }}</text>
      </svg>
      <div class="muted">Next-token loss (nats): the second copy is predicted almost perfectly. It takes two layers: a previous-token head (layer 4, head 11) writes "the token before me" in each stream, and the induction head matches the current token against it, as in the RASP induction program.</div>
    </div>
  </div>
  <div v-else class="muted">{{ status }}</div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'

const presets = [
  { l: 4, h: 11, name: 'previous token' },
  { l: 5, h: 5, name: 'induction' },
  { l: 6, h: 9, name: 'induction' },
  { l: 5, h: 1, name: 'induction' },
  { l: 0, h: 1, name: 'early' },
]
const rep = ref(null)
const status = ref('Loading GPT-2 attention…')
const sel = ref([5, 5])
const hover = ref(-1)
const c = 7, x0 = 46, y0 = 18
const BW = 430, BH = 95
const L = computed(() => (rep.value ? rep.value.length : 0))
const T = computed(() => (rep.value ? rep.value.tokens.length : 0))
const flat = computed(() => rep.value.attn[sel.value[0]][sel.value[1]])
const w = (i, j) => (i < 0 ? 0 : flat.value[(i * (i + 1)) / 2 + j])
const shade = v => `rgba(8, 69, 125, ${Math.min(1, 0.03 + (v / 100) * 1.2)})`
const vis = t => t.replace(/ /g, '·')
const bw = computed(() => BW / (2 * L.value))
const bx = i => i * bw.value
const by = v => (BH - 22) * (1 - v / 15)
const mean = (a, b) => rep.value.loss.slice(a, b).reduce((s, v) => s + v, 0) / (b - a)

onMounted(async () => {
  try {
    const res = await fetch(new URL('../../data/gpt2_attention.json', window.location.origin + import.meta.env.BASE_URL))
    rep.value = (await res.json()).repeat
    hover.value = rep.value.length + 5
  } catch (err) {
    status.value = 'Could not load the GPT-2 attention data.'
  }
})
</script>

<style scoped>
.ih { display: flex; gap: 1.2em; justify-content: center; align-items: flex-start; margin-top: 2px; font-size: 0.6em; }
.ih-left { display: flex; flex-direction: column; gap: 0.4em; }
.ih-heads { display: flex; flex-wrap: wrap; gap: 0.3em; width: 400px; }
.ih-heads button { padding: 0.1em 0.45em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.ih-heads button.on { background: #8c0000; color: #fff; }
.ih-heads button.on .muted { color: #f3d6d6; }
.ih-right { display: flex; flex-direction: column; gap: 0.4em; width: 420px; }
.ih-seq { line-height: 1.6; display: flex; flex-wrap: wrap; row-gap: 2px; }
.wt { font-family: 'Fira Mono', monospace; padding: 1px 2px; margin-right: 1px; }
.wt.q { outline: 2px solid #8c0000; }
.wt.half { margin-left: 0.8em; }
.ax { font-size: 10px; fill: #667678; }
.muted { color: #667678; }
</style>
