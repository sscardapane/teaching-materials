<template>
  <div class="race">
    <svg :width="W" :height="H" class="race-plot" role="img" aria-label="Optimizer paths on a quadratic">
      <ellipse v-for="(c, i) in levels" :key="i" :cx="toX(0)" :cy="toY(0)"
        :rx="Math.sqrt(2 * c) * sx" :ry="Math.sqrt(2 * c / kappa) * sy"
        fill="none" stroke="#c8c8c8" stroke-width="1" />
      <template v-for="m in methods" :key="m.id">
        <g v-if="m.on">
          <polyline :points="pts(m.path)" fill="none" :stroke="m.color" stroke-width="2.4" />
          <circle v-for="(p, i) in visible(m.path)" :key="i" :cx="toX(p[0])" :cy="toY(p[1])" r="3" :fill="m.color" />
        </g>
      </template>
      <text :x="toX(0) + 6" :y="toY(0) - 6" :style="{ fontSize: '16px' }">★</text>
    </svg>

    <div class="race-controls">
      <label>Conditioning κ <b>{{ kappa }}</b>
        <input type="range" min="1" max="50" step="1" v-model.number="kappa" /></label>
      <label>GD / momentum step η <b>{{ eta.toFixed(3) }}</b>
        <input type="range" min="0.005" max="0.15" step="0.005" v-model.number="eta" /></label>
      <div :class="['race-limit', eta < 2 / kappa ? 'ok' : 'bad']">
        stability limit 2/κ = {{ (2 / kappa).toFixed(3) }}:
        {{ eta < 2 / kappa ? 'GD is stable' : 'GD diverges along θ₂' }}
      </div>
      <label>Momentum λ <b>{{ lam.toFixed(2) }}</b>
        <input type="range" min="0" max="0.95" step="0.05" v-model.number="lam" /></label>
      <label>Adam step <b>{{ adamEta.toFixed(2) }}</b>
        <input type="range" min="0.05" max="0.8" step="0.05" v-model.number="adamEta" /></label>
      <table class="race-table">
        <tr><th></th><th>show</th><th>loss after {{ steps }} steps</th></tr>
        <tr v-for="m in methods" :key="m.id">
          <td :style="{ color: m.color }"><b>{{ m.name }}</b></td>
          <td><input type="checkbox" v-model="m.on" /></td>
          <td class="mono">{{ fmt(loss(m.path[m.path.length - 1])) }}</td>
        </tr>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'

const W = 470, H = 330
const xMin = -5.5, xMax = 1.5, yMin = -2.2, yMax = 2.2
const sx = W / (xMax - xMin), sy = H / (yMax - yMin)
const toX = x => (x - xMin) * sx
const toY = y => H - (y - yMin) * sy

const kappa = ref(20)
const eta = ref(0.095)
const lam = ref(0.7)
const adamEta = ref(0.3)
const steps = 20
const start = [-4.5, 1.0]
const levels = [0.05, 0.2, 0.6, 1.5, 3, 6, 10.2]

const grad = p => [p[0], kappa.value * p[1]]
const loss = p => 0.5 * (p[0] ** 2 + kappa.value * p[1] ** 2)

function heavyBall(step, beta) {
  let p = [...start], u = [0, 0]
  const out = [[...p]]
  for (let t = 0; t < steps; t++) {
    const g = grad(p)
    u = [beta * u[0] - step * g[0], beta * u[1] - step * g[1]]
    p = [p[0] + u[0], p[1] + u[1]]
    out.push([...p])
  }
  return out
}

function adam(step) {
  let p = [...start], m = [0, 0], v = [0, 0]
  const b1 = 0.9, b2 = 0.999, out = [[...p]]
  for (let t = 1; t <= steps; t++) {
    const g = grad(p)
    m = m.map((mi, i) => b1 * mi + (1 - b1) * g[i])
    v = v.map((vi, i) => b2 * vi + (1 - b2) * g[i] * g[i])
    p = p.map((pi, i) => pi - step * (m[i] / (1 - b1 ** t)) / (Math.sqrt(v[i] / (1 - b2 ** t)) + 1e-8))
    out.push([...p])
  }
  return out
}

const shown = reactive({ gd: true, mom: true, adam: true })
const methods = computed(() => [
  { id: 'gd', name: 'GD', color: '#B85450', path: heavyBall(eta.value, 0), get on() { return shown.gd }, set on(v) { shown.gd = v } },
  { id: 'mom', name: 'Momentum', color: '#6C8EBF', path: heavyBall(eta.value * (1 - lam.value), lam.value), get on() { return shown.mom }, set on(v) { shown.mom = v } },
  { id: 'adam', name: 'Adam', color: '#5d9a43', path: adam(adamEta.value), get on() { return shown.adam }, set on(v) { shown.adam = v } },
])

const inside = p => Number.isFinite(p[0]) && p[0] > xMin - 2 && p[0] < xMax + 2 && p[1] > yMin - 2 && p[1] < yMax + 2
function visible(path) {
  const out = []
  for (const p of path) { if (!inside(p)) break; out.push(p) }
  return out
}
const pts = path => visible(path).map(p => `${toX(p[0])},${toY(p[1])}`).join(' ')
const fmt = v => (!Number.isFinite(v) || v > 1e6 ? 'diverged' : v < 1e-4 ? v.toExponential(1) : v.toFixed(4))
</script>

<style scoped>
.race { display: flex; gap: 1.6em; align-items: center; justify-content: center; margin-top: 6px; }
.race-plot { border: 1px solid #9ba8aa; background: #fff; }
.race-controls { display: flex; flex-direction: column; gap: 0.38em; width: 330px; font-size: 0.7em; }
.race-controls label { display: grid; grid-template-columns: 1fr auto; gap: 0.1em 0.6em; }
.race-controls input[type=range] { grid-column: 1 / -1; width: 100%; accent-color: #8c0000; }
.race-limit { padding-left: 0.5em; border-left: 4px solid #617b67; font-weight: 600; }
.race-limit.bad { border-color: #8c0000; color: #8c0000; }
.race-table { border-collapse: collapse; margin-top: 0.3em; }
.race-table th { font-weight: 400; color: #667678; text-align: left; padding-right: 0.8em; }
.race-table td { padding: 0.08em 0.8em 0.08em 0; }
.mono { font-family: 'Fira Mono', monospace; }
</style>
