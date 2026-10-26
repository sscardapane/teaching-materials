<template>
  <div class="blind">
    <svg :width="W" :height="H" class="blind-plot" role="img" aria-label="A hidden loss curve explored one evaluation at a time">
      <rect v-if="!revealed" x="0" y="0" :width="W" :height="H" fill="#eef0f0" />
      <text v-if="!revealed && visits.length === 1" :x="W / 2" :y="H / 2" text-anchor="middle" class="fog">the loss is hidden: you only see where you have been</text>
      <polyline v-if="revealed" :points="curve" fill="none" stroke="#999" stroke-width="2" />
      <template v-if="revealed && showGD">
        <polyline :points="gdPoints" fill="none" stroke="#6C8EBF" stroke-width="2" stroke-dasharray="4 3" />
        <circle v-for="(x, i) in gdVisible" :key="'g' + i" :cx="toX(x)" :cy="toY(f(x))" r="3.5" fill="#6C8EBF" />
      </template>
      <polyline :points="visitPoints" fill="none" stroke="#8c0000" stroke-width="1.5" stroke-dasharray="2 3" />
      <g v-for="(x, i) in visits" :key="i">
        <line :x1="toX(x - 0.35)" :y1="toY(f(x) - 0.35 * df(x))" :x2="toX(x + 0.35)" :y2="toY(f(x) + 0.35 * df(x))" stroke="#8c0000" stroke-width="2" />
        <circle :cx="toX(x)" :cy="toY(f(x))" :r="i === visits.length - 1 ? 5.5 : 3.5" :fill="i === visits.length - 1 ? '#d79b00' : '#8c0000'" />
      </g>
      <text v-if="revealed" :x="toX(xStar) - 6" :y="toY(f(xStar)) + 18" :style="{ fontSize: '16px' }">★</text>
    </svg>

    <div class="blind-controls">
      <div class="blind-readout mono">
        x = {{ current.toFixed(2) }}<br />
        loss f(x) = {{ f(current).toFixed(3) }}<br />
        slope f′(x) = <b :class="df(current) > 0 ? 'up' : 'down'">{{ df(current) > 0 ? '+' : '' }}{{ df(current).toFixed(3) }}</b>
      </div>

      <label>Move by Δx <b>{{ (dx > 0 ? '+' : '') + dx.toFixed(1) }}</b>
        <input type="range" min="-3" max="3" step="0.1" v-model.number="dx" /></label>
      <div class="blind-buttons">
        <button @click="move" :disabled="outside">Go</button>
        <span class="muted">evaluations: <b>{{ visits.length }}</b></span>
      </div>
      <div v-if="found" class="blind-found">Global minimum reached in {{ visits.length - 1 }} moves.</div>
      <div v-else-if="outside" class="blind-warn">That would leave the interval [0, 10].</div>

      <div class="blind-buttons">
        <button @click="revealed = !revealed">{{ revealed ? 'Hide' : 'Reveal' }}</button>
        <button @click="newLandscape">New landscape</button>
      </div>
      <template v-if="revealed">
        <label><span><input type="checkbox" v-model="showGD" /> GD from the same start, η <b>{{ eta.toFixed(2) }}</b></span>
          <input type="range" min="0.05" max="3" step="0.05" v-model.number="eta" /></label>
        <div class="muted">GD after 30 steps: loss {{ f(gdPath[gdPath.length - 1]).toFixed(3) }}
          (minimum {{ f(xStar).toFixed(3) }})</div>
      </template>
    </div>
  </div>
</template>

<script setup>
// Written by Claude (2026-10-26): a guessing game for 2c. Students see only
// the loss and slope at the points they have visited, like an optimizer.
import { computed, ref } from 'vue'

const W = 520, H = 330
const seed = ref(7)
const dx = ref(1)
const eta = ref(0.5)
const showGD = ref(true)
const revealed = ref(false)

function rng(s) {
  return () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
}

const params = computed(() => {
  const r = rng(seed.value)
  return {
    c: 3 + 4 * r(),
    waves: [0, 1, 2].map(k => ({ a: 0.25 + 0.45 * r(), w: 1 + 0.9 * k + 0.6 * r(), p: 6.28 * r() })),
    start: r() < 0.5 ? 0.6 : 9.4,
  }
})
const f = x => 0.09 * (x - params.value.c) ** 2 + params.value.waves.reduce((s, q) => s + q.a * Math.sin(q.w * x + q.p), 0)
const df = x => 0.18 * (x - params.value.c) + params.value.waves.reduce((s, q) => s + q.a * q.w * Math.cos(q.w * x + q.p), 0)

const grid = computed(() => Array.from({ length: 1001 }, (_, i) => i / 100))
const xStar = computed(() => grid.value.reduce((b, x) => (f(x) < f(b) ? x : b), 0))
const yRange = computed(() => {
  const ys = grid.value.map(f)
  const lo = Math.min(...ys), hi = Math.max(...ys)
  return [lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo)]
})
const toX = x => x / 10 * W
const toY = y => H - (y - yRange.value[0]) / (yRange.value[1] - yRange.value[0]) * H

const visits = ref([])
const current = computed(() => visits.value[visits.value.length - 1])
const outside = computed(() => current.value + dx.value < 0 || current.value + dx.value > 10)
const found = computed(() => visits.value.some(x => f(x) - f(xStar.value) < 0.03))

function reset() { visits.value = [params.value.start]; revealed.value = false }
function newLandscape() { seed.value += 1; reset() }
function move() { if (!outside.value) visits.value = [...visits.value, current.value + dx.value] }
reset()

const curve = computed(() => grid.value.filter((_, i) => i % 4 === 0).map(x => `${toX(x)},${toY(f(x))}`).join(' '))
const visitPoints = computed(() => visits.value.map(x => `${toX(x)},${toY(f(x))}`).join(' '))
const gdPath = computed(() => {
  const out = [params.value.start]
  for (let t = 0; t < 30; t++) out.push(out[t] - eta.value * df(out[t]))
  return out
})
const gdVisible = computed(() => {
  const out = []
  for (const x of gdPath.value) { if (!(x >= 0 && x <= 10)) break; out.push(x) }
  return out
})
const gdPoints = computed(() => gdVisible.value.map(x => `${toX(x)},${toY(f(x))}`).join(' '))
</script>

<style scoped>
.blind { display: flex; gap: 1.8em; align-items: center; justify-content: center; margin-top: 6px; }
.blind-plot { border: 1px solid #9ba8aa; background: #fff; }
.fog { fill: #667678; font-size: 14px; }
.blind-controls { display: flex; flex-direction: column; gap: 0.5em; width: 330px; font-size: 0.7em; }
.blind-controls label { display: grid; grid-template-columns: 1fr auto; gap: 0.1em 0.6em; }
.blind-controls input[type=range] { grid-column: 1 / -1; width: 100%; accent-color: #8c0000; }
.blind-readout { padding: 0.3em 0.6em; border-left: 4px solid #8c0000; line-height: 1.5; }
.up { color: #8c0000; }
.down { color: #5d9a43; }
.blind-buttons { display: flex; gap: 0.5em; align-items: center; }
.blind-buttons button { padding: 0.2em 0.7em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.blind-buttons button:disabled { opacity: 0.4; cursor: default; }
.blind-found { font-weight: 600; color: #5d9a43; }
.blind-warn { color: #8c0000; }
.muted { color: #667678; }
.mono { font-family: 'Fira Mono', monospace; }
</style>
