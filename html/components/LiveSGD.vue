<template>
  <div class="sgd">
    <svg :width="W" :height="H" class="sgd-plot" role="img" aria-label="Logistic regression trained with mini-batch SGD">
      <polygon :points="halfPlane" fill="#f8e9e8" />
      <circle v-for="(p, i) in data" :key="i" :cx="toX(p.x[0])" :cy="toY(p.x[1])" r="3.6"
        :fill="p.y ? '#B85450' : '#6C8EBF'" :stroke="inBatch.has(i) ? '#111' : 'none'" stroke-width="2" />
      <line v-if="line" :x1="line[0]" :y1="line[1]" :x2="line[2]" :y2="line[3]" stroke="#8c0000" stroke-width="2.5" />
    </svg>

    <div class="sgd-right">
      <svg :width="LW" :height="LH" class="sgd-loss" role="img" aria-label="Training loss curves">
        <polyline v-for="(r, i) in ghosts" :key="i" :points="lossPoints(r)" fill="none" :stroke="ghostColors[i]" stroke-width="1.6" />
        <polyline :points="lossPoints(run)" fill="none" stroke="#8c0000" stroke-width="2.2" />
        <text x="6" y="14" class="axis">full-data loss (0.2 to 1)</text>
        <text :x="LW - 6" :y="LH - 6" text-anchor="end" class="axis">{{ perExample ? 'examples seen' : 'updates' }}</text>
      </svg>
      <div class="sgd-legend">
        <span class="key" style="color: #8c0000">B = {{ run.B }}, η = {{ run.eta }}</span>
        <span v-for="(r, i) in ghosts" :key="i" class="key" :style="{ color: ghostColors[i] }">B = {{ r.B }}, η = {{ r.eta }}</span>
      </div>

      <div class="sgd-controls">
        <div class="sgd-row">batch size
          <button v-for="b in sizes" :key="b" :class="{ on: b === B }" @click="setB(b)">{{ b === N ? 'all (' + N + ')' : b }}</button>
        </div>
        <label>Learning rate η <b>{{ eta.toFixed(2) }}</b>
          <input type="range" min="0.05" max="2" step="0.05" v-model.number="eta" @change="restart" /></label>
        <div class="sgd-row">
          <button @click="toggle">{{ playing ? 'Pause' : 'Play' }}</button>
          <button @click="step">Step</button>
          <button @click="restart">Restart</button>
          <label class="inline"><input type="checkbox" v-model="perExample" /> x-axis: examples seen</label>
        </div>
        <div class="mono muted">update {{ run.loss.length - 1 }} · loss {{ run.loss[run.loss.length - 1].toFixed(3) }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
// Written by Claude (2026-10-26): logistic regression on 2D data, trained
// live. Mini-batch noise is visible on the boundary and on the loss curve.
import { computed, onBeforeUnmount, ref } from 'vue'

const W = 400, H = 340, LW = 380, LH = 170
const box = [-4, 4, -3.4, 3.4]
const N = 200
const sizes = [1, 8, 32, N]
const MAX_UPDATES = 300
const ghostColors = ['#9ba8aa', '#c4a35a', '#7aa874']

function rng(s) {
  return () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
}
const r = rng(11)
const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r())
const data = Array.from({ length: N }, (_, i) => {
  const y = i % 2
  const c = y ? [1.1, 0.7] : [-1.1, -0.7]
  return { x: [c[0] + 1.0 * gauss(), c[1] + 1.0 * gauss()], y }
})

const sigmoid = z => 1 / (1 + Math.exp(-z))
function fullLoss(w) {
  let s = 0
  for (const p of data) {
    const z = w[0] * p.x[0] + w[1] * p.x[1] + w[2]
    s += Math.log(1 + Math.exp(-Math.abs(z))) + Math.max(z, 0) - p.y * z
  }
  return s / N
}

const B = ref(8), eta = ref(0.5), perExample = ref(false), playing = ref(false)
const w0 = [-1.5, 2.0, 1.0]
const run = ref(null)
const inBatch = ref(new Set())
const ghosts = ref([])
let order = [], cursor = 0, sampler = rng(3)

function shuffle() {
  order = [...Array(N).keys()]
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(sampler() * (i + 1)); [order[i], order[j]] = [order[j], order[i]] }
  cursor = 0
}
function fresh() {
  sampler = rng(3); shuffle()
  run.value = { B: B.value, eta: eta.value, w: [...w0], loss: [fullLoss(w0)] }
  inBatch.value = new Set()
}
function restart() {
  if (run.value && run.value.loss.length > 2) ghosts.value = [run.value, ...ghosts.value].slice(0, 3)
  stop(); fresh()
}
function setB(b) { B.value = b; restart() }

function step() {
  const R = run.value
  if (R.loss.length > MAX_UPDATES) { stop(); return }
  if (cursor + R.B > N) shuffle()
  const idx = order.slice(cursor, cursor + R.B); cursor += R.B
  const g = [0, 0, 0]
  for (const i of idx) {
    const p = data[i]
    const e = sigmoid(R.w[0] * p.x[0] + R.w[1] * p.x[1] + R.w[2]) - p.y
    g[0] += e * p.x[0]; g[1] += e * p.x[1]; g[2] += e
  }
  R.w = R.w.map((wi, k) => wi - R.eta * g[k] / idx.length)
  R.loss = [...R.loss, fullLoss(R.w)]
  inBatch.value = new Set(idx)
}

let timer = null
function stop() { playing.value = false; if (timer) clearInterval(timer); timer = null }
function toggle() {
  if (playing.value) { stop(); return }
  playing.value = true
  timer = setInterval(step, 60)
}
onBeforeUnmount(stop)
fresh()

const toX = x => (x - box[0]) / (box[1] - box[0]) * W
const toY = y => H - (y - box[2]) / (box[3] - box[2]) * H

// Decision line w0 x + w1 y + b = 0, clipped to the box, and the class-1 half plane.
function clip(w) {
  const pts = []
  const [a, b, c] = w
  if (Math.abs(b) > 1e-9) for (const x of [box[0], box[1]]) { const y = -(a * x + c) / b; if (y >= box[2] && y <= box[3]) pts.push([x, y]) }
  if (Math.abs(a) > 1e-9) for (const y of [box[2], box[3]]) { const x = -(b * y + c) / a; if (x >= box[0] && x <= box[1]) pts.push([x, y]) }
  return pts
}
const line = computed(() => {
  const p = clip(run.value.w)
  return p.length >= 2 ? [toX(p[0][0]), toY(p[0][1]), toX(p[1][0]), toY(p[1][1])] : null
})
const halfPlane = computed(() => {
  const [a, b, c] = run.value.w
  const corners = [[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]]
  const poly = []
  for (let i = 0; i < 4; i++) {
    const p = corners[i], q = corners[(i + 1) % 4]
    const sp = a * p[0] + b * p[1] + c, sq = a * q[0] + b * q[1] + c
    if (sp >= 0) poly.push(p)
    if ((sp >= 0) !== (sq >= 0)) { const t = sp / (sp - sq); poly.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]) }
  }
  return poly.map(p => `${toX(p[0])},${toY(p[1])}`).join(' ')
})

// The start loss is far above the rest; zoom on the range where B matters.
const yLo = 0.2, yHi = 1.0
function lossPoints(R) {
  const xMax = perExample.value ? MAX_UPDATES * 32 : MAX_UPDATES
  return R.loss
    .map((l, t) => [perExample.value ? t * R.B : t, l])
    .filter(([x]) => x <= xMax)
    .map(([x, l]) => `${x / xMax * LW},${LH - 4 - Math.min(Math.max((l - yLo) / (yHi - yLo), 0), 1) * (LH - 24)}`).join(' ')
}
</script>

<style scoped>
.sgd { display: flex; gap: 1.4em; align-items: center; justify-content: center; margin-top: 4px; }
.sgd-plot, .sgd-loss { border: 1px solid #9ba8aa; background: #fff; }
.sgd-right { display: flex; flex-direction: column; gap: 0.35em; width: 380px; font-size: 0.68em; }
.axis { fill: #667678; font-size: 11px; }
.sgd-legend { display: flex; gap: 1em; flex-wrap: wrap; }
.key { font-weight: 600; }
.sgd-controls { display: flex; flex-direction: column; gap: 0.4em; }
.sgd-controls label { display: grid; grid-template-columns: 1fr auto; gap: 0.1em 0.6em; }
.sgd-controls label.inline { display: inline; color: #667678; }
.sgd-controls input[type=range] { grid-column: 1 / -1; width: 100%; accent-color: #8c0000; }
.sgd-row { display: flex; gap: 0.4em; align-items: center; flex-wrap: wrap; }
.sgd-row button { padding: 0.15em 0.6em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.sgd-row button.on { background: #8c0000; color: #fff; }
.muted { color: #667678; }
.mono { font-family: 'Fira Mono', monospace; }
</style>
