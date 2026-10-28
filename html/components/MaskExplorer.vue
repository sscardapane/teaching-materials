<!-- Written by Claude (2026-10-28): attention weights of one head under
     different masks and positional biases, for the Transformer demos.
     Scores are either all equal (only the mask and the bias matter) or come
     from fixed random queries and keys (content plus position). -->
<template>
  <div class="mx">
    <div class="mx-side">
    <div class="mx-controls">
      <div class="mx-group">
        <span class="muted">scores</span>
        <button v-for="m in scoreModes" :key="m.id" :class="{ on: scores === m.id }" @click="scores = m.id">{{ m.label }}</button>
      </div>
      <label class="mx-check"><input type="checkbox" v-model="causal" /> causal mask</label>
      <label class="mx-check"><input type="checkbox" v-model="alibi" /> ALiBi</label>
      <label v-if="alibi">slope <b>{{ slopeLabel }}</b>
        <input type="range" min="-5" max="1" step="1" v-model.number="logSlope" /></label>
      <label v-if="scores === 'content'">sharpness <b>{{ sharp.toFixed(1) }}</b>
        <input type="range" min="0" max="4" step="0.5" v-model.number="sharp" /></label>
    </div>
    <div class="mx-readout">
      average distance attended by the last token: <b>{{ lastDistance.toFixed(1) }}</b> tokens
      <span class="muted">(uniform: {{ uniformDistance.toFixed(1) }})</span>
    </div>
    </div>

    <svg :width="SW" :height="SH" role="img" aria-label="Attention matrix">
      <text v-for="(t, j) in tokens" :key="'c' + j" :x="x0 + j * c + c / 2" :y="y0 - 6" text-anchor="start" class="tk"
        :transform="`rotate(-40 ${x0 + j * c + c / 2} ${y0 - 6})`">{{ t }}</text>
      <g v-for="(row, i) in A" :key="'r' + i" @mouseenter="hover = i" @mouseleave="hover = -1">
        <text :x="x0 - 8" :y="y0 + i * c + c / 2 + 5" text-anchor="end" class="tk" :class="{ hl: hover === i }">{{ tokens[i] }}</text>
        <rect v-for="(w, j) in row" :key="j" :x="x0 + j * c" :y="y0 + i * c" :width="c" :height="c"
          :fill="masked(i, j) ? '#fbecec' : shade(w)" stroke="#c9d1d3" stroke-width="0.6" />
        <text v-for="(w, j) in row" :key="'v' + j" :x="x0 + j * c + c / 2" :y="y0 + i * c + c / 2 + 4"
          text-anchor="middle" class="val" :fill="masked(i, j) ? '#b85450' : (w > 0.45 ? '#fff' : '#333')">{{ masked(i, j) ? '−∞' : w.toFixed(2) }}</text>
      </g>
      <rect v-if="hover >= 0" :x="x0" :y="y0 + hover * c" :width="T * c" :height="c" fill="none" stroke="#8c0000" stroke-width="2" />
    </svg>


  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const tokens = ['the', 'cat', 'that', 'my', 'mother', 'had', 'sat', 'on', 'the', 'mat']
const T = tokens.length
const c = 27, x0 = 60, y0 = 50
const SW = x0 + T * c + 40, SH = y0 + T * c + 4
const scoreModes = [{ id: 'equal', label: 'all equal' }, { id: 'content', label: 'random q, k' }]
const scores = ref('equal')
const causal = ref(true)
const alibi = ref(false)
const logSlope = ref(-1)
const sharp = ref(2)
const hover = ref(-1)

// Fixed pseudo-random queries and keys (D_k = 4), the same at every reload.
let seed = 7
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1
const Q = tokens.map(() => [rnd(), rnd(), rnd(), rnd()])
const K = tokens.map(() => [rnd(), rnd(), rnd(), rnd()])

const slope = computed(() => 2 ** logSlope.value)
const slopeLabel = computed(() => (logSlope.value >= 0 ? String(slope.value) : '1/' + 2 ** -logSlope.value))
const masked = (i, j) => causal.value && j > i

const A = computed(() => tokens.map((_, i) => {
  const s = tokens.map((_, j) => {
    if (masked(i, j)) return -Infinity
    let v = scores.value === 'content' ? sharp.value * Q[i].reduce((a, q, d) => a + q * K[j][d], 0) : 0
    if (alibi.value) v -= slope.value * Math.abs(i - j)
    return v
  })
  const m = Math.max(...s)
  const e = s.map(v => (v === -Infinity ? 0 : Math.exp(v - m)))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map(v => v / z)
}))

const shade = w => `rgba(8, 69, 125, ${Math.min(1, 0.04 + w * 1.25)})`
const lastDistance = computed(() => A.value[T - 1].reduce((a, w, j) => a + w * (T - 1 - j), 0))
// The last token sees all T tokens with or without the causal mask.
const uniformDistance = computed(() => (T - 1) / 2)
</script>

<style scoped>
.mx { display: flex; flex-direction: row-reverse; justify-content: center; align-items: center; gap: 2em; margin-top: 4px; }
.mx-side { display: flex; flex-direction: column; gap: 1em; width: 300px; }
.mx-controls { display: flex; flex-direction: column; gap: 0.8em; align-items: flex-start; font-size: 0.66em; }
.mx-controls label { display: flex; flex-direction: column; min-width: 110px; }
.mx-controls input[type=range] { accent-color: #8c0000; }
.mx-check { flex-direction: row !important; gap: 0.4em; align-items: center; min-width: 0 !important; }
.mx-group { display: flex; gap: 0.3em; align-items: center; }
.mx-group button { padding: 0.15em 0.55em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.mx-group button.on { background: #8c0000; color: #fff; }
.mx-readout { font-size: 0.66em; }
.tk { font-family: 'Fira Mono', monospace; font-size: 12px; fill: #333; }
.tk.hl { fill: #8c0000; font-weight: bold; }
.val { font-size: 8.5px; font-family: 'Fira Sans', sans-serif; }
.muted { color: #667678; }
</style>
