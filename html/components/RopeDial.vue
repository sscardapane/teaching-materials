<!-- Written by Claude (2026-10-28): RoPE for the Transformer demos. Left: one
     pair of components of a query at position t and a key at position t',
     each rotated by its position. Right: the score summed over all pairs
     (q = k, D_k = 32, base b), which depends only on the distance t - t'. -->
<template>
  <div class="rope">
    <div class="rope-left">
      <svg :width="R2" :height="R2" role="img" aria-label="Rotated query and key">
        <circle :cx="C" :cy="C" :r="r" fill="none" stroke="#c9d1d3" />
        <line :x1="C - r - 10" :y1="C" :x2="C + r + 10" :y2="C" stroke="#e1e5e6" />
        <line :x1="C" :y1="C - r - 10" :x2="C" :y2="C + r + 10" stroke="#e1e5e6" />
        <path :d="arc" fill="none" stroke="#6C8EBF" stroke-width="2" />
        <line :x1="C" :y1="C" :x2="px(qAng)" :y2="py(qAng)" stroke="#7a4f8e" stroke-width="3" marker-end="url(#hq)" />
        <line :x1="C" :y1="C" :x2="px(kAng)" :y2="py(kAng)" stroke="#b78400" stroke-width="3" marker-end="url(#hk)" />
        <text :x="px(qAng, 1.17)" :y="py(qAng, 1.17) + 5" text-anchor="middle" class="lab q">q̃ (t={{ t }})</text>
        <text :x="px(kAng, 1.17)" :y="py(kAng, 1.17) + 5" text-anchor="middle" class="lab k">k̃ (t'={{ tp }})</text>
        <defs>
          <marker id="hq" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#7a4f8e" /></marker>
          <marker id="hk" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#b78400" /></marker>
        </defs>
      </svg>
      <div class="rope-controls">
        <label>query position t <b>{{ t }}</b><input type="range" min="0" max="40" v-model.number="t" /></label>
        <label>key position t' <b>{{ tp }}</b><input type="range" min="0" max="40" v-model.number="tp" /></label>
        <label>rotation per position <b>{{ theta }}°</b><input type="range" min="2" max="40" v-model.number="theta" /></label>
        <div class="rope-buttons">
          <button @click="shift(1)">shift both +1</button>
          <button @click="shift(-1)">shift both −1</button>
        </div>
        <div class="readout">angle between them: <b>{{ angle.toFixed(0) }}°</b>, pair score <b>{{ pairScore.toFixed(2) }}</b></div>
      </div>
    </div>

    <div class="rope-right">
      <svg :width="PW" :height="PH" role="img" aria-label="Score against distance">
        <line :x1="pl" :y1="yOf(0)" :x2="PW - 8" :y2="yOf(0)" stroke="#c9d1d3" />
        <line :x1="pl" :y1="8" :x2="pl" :y2="PH - 26" stroke="#9ba8aa" />
        <path :d="curve" fill="none" stroke="#08457d" stroke-width="2" />
        <circle :cx="xOf(dist)" :cy="yOf(fullScore(dist))" r="6" fill="#8c0000" />
        <text v-for="d in [0, 20, 40, 60, 80]" :key="d" :x="xOf(d)" :y="PH - 10" text-anchor="middle" class="ax">{{ d }}</text>
        <text :x="PW - 8" :y="yOf(0) - 6" text-anchor="end" class="ax">distance |t − t'|</text>
        
        <text :x="pl + 6" :y="18" class="ax">score (all pairs, q = k)</text>
      </svg>
      <label class="base">base b <b>{{ base.toLocaleString('en') }}</b>
        <input type="range" min="1" max="5" step="1" v-model.number="logBase" /></label>
      <div class="note">Shifting both tokens moves both arrows but not the red dot: the score
        depends only on the distance. Nearby tokens score higher, with ripples from the fast pairs.</div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const R2 = 250, C = 125, r = 82
const t = ref(3), tp = ref(1), theta = ref(25), logBase = ref(4)
const Q0 = 10, K0 = 100          // angles of the unrotated query and key pair
const qAng = computed(() => Q0 + theta.value * t.value)
const kAng = computed(() => K0 + theta.value * tp.value)
const px = (a, s = 1) => C + s * r * Math.cos((a * Math.PI) / 180)
const py = (a, s = 1) => C - s * r * Math.sin((a * Math.PI) / 180)
const angle = computed(() => {
  const d = (((kAng.value - qAng.value) % 360) + 360) % 360
  return d > 180 ? 360 - d : d
})
const pairScore = computed(() => Math.cos((angle.value * Math.PI) / 180))
const arc = computed(() => {
  const a0 = qAng.value, d = (((kAng.value - a0) % 360) + 540) % 360 - 180
  const rr = 36, x0 = C + rr * Math.cos((a0 * Math.PI) / 180), y0 = C - rr * Math.sin((a0 * Math.PI) / 180)
  const a1 = a0 + d, x1 = C + rr * Math.cos((a1 * Math.PI) / 180), y1 = C - rr * Math.sin((a1 * Math.PI) / 180)
  return `M${x0},${y0} A${rr},${rr} 0 0 ${d > 0 ? 0 : 1} ${x1},${y1}`
})
function shift(s) {
  if (Math.min(t.value, tp.value) + s < 0 || Math.max(t.value, tp.value) + s > 40) return
  t.value += s; tp.value += s
}

// Full RoPE score with q = k = all ones, D_k = 32: sum_j 2 cos(w_j d), w_j = b^(-2j/D_k).
const Dk = 32
const base = computed(() => 10 ** logBase.value)
const fullScore = d => {
  let s = 0
  for (let j = 0; j < Dk / 2; j++) s += 2 * Math.cos(d * base.value ** (-2 * j / Dk))
  return s / Dk
}
const dist = computed(() => Math.abs(t.value - tp.value))
const PW = 380, PH = 230, pl = 30, maxD = 80
const xOf = d => pl + (d / maxD) * (PW - pl - 12)
const yOf = v => 14 + (1 - v) / 1.2 * (PH - 50)
const curve = computed(() => {
  base.value
  let p = ''
  for (let d = 0; d <= maxD; d += 0.25) p += (d ? 'L' : 'M') + xOf(d) + ',' + yOf(fullScore(d))
  return p
})
</script>

<style scoped>
.rope { display: flex; gap: 2.2em; justify-content: center; align-items: flex-start; margin-top: 4px; }
.rope-left { display: flex; gap: 1em; align-items: center; }
.rope-controls { display: flex; flex-direction: column; gap: 0.5em; font-size: 0.6em; width: 210px; }
.rope-controls label, .base { display: flex; flex-direction: column; }
input[type=range] { accent-color: #8c0000; }
.rope-buttons { display: flex; gap: 0.4em; }
.rope-buttons button { white-space: nowrap; padding: 0.15em 0.5em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.rope-right { display: flex; flex-direction: column; gap: 0.4em; font-size: 0.6em; width: 380px; }
.readout { line-height: 1.4; }
.note { color: #445; line-height: 1.35; }
.lab { font-size: 13px; font-family: 'Fira Sans', sans-serif; }
.lab.q { fill: #7a4f8e; } .lab.k { fill: #b78400; }
.ax { font-size: 11px; fill: #667678; font-family: 'Fira Sans', sans-serif; }
</style>
