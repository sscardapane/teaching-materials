<template>
  <div class="race">
    <div class="race-left">
      <div class="race-surfaces">
        <button v-for="s in surfaces" :key="s.id" :class="{ on: s.id === surfaceId }" @click="setSurface(s.id)">{{ s.name }}</button>
      </div>
      <svg :width="W" :height="H" class="race-plot" role="img" aria-label="Optimizer paths on a loss surface" @click="onClick">
        <image v-if="background" :href="background" x="0" y="0" :width="W" :height="H" preserveAspectRatio="none" />
        <template v-for="m in methods" :key="m.id">
          <g v-if="m.on">
            <polyline :points="pts(m.path)" fill="none" :stroke="m.color" stroke-width="2.4" />
            <circle v-for="(p, i) in visible(m.path)" :key="i" :cx="toX(p[0])" :cy="toY(p[1])" r="2.6" :fill="m.color" />
          </g>
        </template>
        <text v-for="(m, i) in surface.minima" :key="i" :x="toX(m[0]) - 6" :y="toY(m[1]) + 6" :style="{ fontSize: '16px' }">★</text>
        <circle :cx="toX(start[0])" :cy="toY(start[1])" r="5" fill="none" stroke="#333" stroke-width="1.5" />
      </svg>
      <div class="race-caption">{{ surface.caption }} <span class="muted">Click the plot to move the start.</span></div>
    </div>

    <div class="race-controls">
      <label v-if="surface.id === 'valley'">Conditioning κ <b>{{ kappa }}</b>
        <input type="range" min="1" max="50" step="1" v-model.number="kappa" /></label>
      <label>GD / momentum step η <b>{{ eta.toFixed(3) }}</b>
        <input type="range" :min="surface.eta[0]" :max="surface.eta[1]" :step="surface.eta[0]" v-model.number="eta" /></label>
      <div v-if="surface.id === 'valley'" :class="['race-limit', eta < 2 / kappa ? 'ok' : 'bad']">
        stability limit 2/κ = {{ (2 / kappa).toFixed(3) }}:
        {{ eta < 2 / kappa ? 'GD is stable' : 'GD diverges along θ₂' }}
      </div>
      <label>Momentum λ <b>{{ lam.toFixed(2) }}</b>
        <input type="range" min="0" max="0.95" step="0.05" v-model.number="lam" /></label>
      <label>Adam step <b>{{ adamEta.toFixed(2) }}</b>
        <input type="range" min="0.01" max="0.8" step="0.01" v-model.number="adamEta" /></label>
      <label>Steps <b>{{ steps }}</b>
        <input type="range" min="5" max="300" step="5" v-model.number="steps" /></label>
      <table class="race-table">
        <tr><th></th><th>show</th><th>loss gap after {{ steps }} steps</th></tr>
        <tr v-for="m in methods" :key="m.id">
          <td :style="{ color: m.color }"><b>{{ m.name }}</b></td>
          <td><input type="checkbox" v-model="m.on" /></td>
          <td class="mono">{{ fmt(gap(m.path[m.path.length - 1])) }}</td>
        </tr>
      </table>
    </div>
  </div>
</template>

<script setup>
// Written by Claude (2026-10-26): the 14 Oct valley race, extended with a
// saddle, the Rosenbrock banana and a bumpy surface, a step count and a
// movable start.
import { computed, onMounted, reactive, ref, watch } from 'vue'

const W = 470, H = 300
const kappa = ref(20)

const surfaces = [
  {
    id: 'valley', name: 'valley', box: [-5.5, 1.5, -2.2, 2.2], start: [-4.5, 1.0],
    eta: [0.005, 0.15], defaults: { eta: 0.095, lam: 0.7, adam: 0.3, steps: 20 },
    caption: 'A quadratic valley. Raise κ: GD diverges once η > 2/κ.',
    f: (x, y) => 0.5 * (x * x + kappa.value * y * y),
    g: (x, y) => [x, kappa.value * y],
    minima: [[0, 0]], fmin: 0,
  },
  {
    id: 'saddle', name: 'saddle', box: [-5.5, 1.5, -2.2, 2.2], start: [-4.5, 0.001],
    eta: [0.01, 0.5], defaults: { eta: 0.1, lam: 0.9, adam: 0.1, steps: 70 },
    caption: 'A saddle at the origin, minima at θ₂ = ±1. Starting almost on the ridge, who escapes first?',
    f: (x, y) => 0.5 * x * x + 0.25 * y ** 4 - 0.5 * y * y,
    g: (x, y) => [x, y ** 3 - y],
    minima: [[0, 1], [0, -1]], fmin: -0.25,
  },
  {
    id: 'banana', name: 'banana', box: [-2, 2, -1, 3], start: [-1.5, 2.5],
    eta: [0.001, 0.05], defaults: { eta: 0.015, lam: 0.9, adam: 0.1, steps: 200 },
    caption: 'The Rosenbrock banana: reaching the valley is easy, following it to the minimum is not.',
    f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
    minima: [[1, 1]], fmin: 0,
  },
  {
    id: 'bumpy', name: 'bumpy', box: [-4, 4, -2.8, 2.8], start: [-3.4, 2.3],
    eta: [0.01, 0.6], defaults: { eta: 0.2, lam: 0.9, adam: 0.1, steps: 100 },
    caption: 'A bowl covered in bumps: many local minima, one global minimum at the centre.',
    f: (x, y) => 0.15 * (x * x + y * y) + 0.3 * (2 - Math.cos(3 * x) - Math.cos(3 * y)),
    g: (x, y) => [0.3 * x + 0.9 * Math.sin(3 * x), 0.3 * y + 0.9 * Math.sin(3 * y)],
    minima: [[0, 0]], fmin: 0,
  },
]

const surfaceId = ref('valley')
const surface = computed(() => surfaces.find(s => s.id === surfaceId.value))
const eta = ref(0.095), lam = ref(0.7), adamEta = ref(0.3), steps = ref(20)
const start = ref([-4.5, 1.0])

function setSurface(id) {
  surfaceId.value = id
  const s = surface.value
  eta.value = s.defaults.eta; lam.value = s.defaults.lam
  adamEta.value = s.defaults.adam; steps.value = s.defaults.steps
  start.value = [...s.start]
}

const sx = computed(() => W / (surface.value.box[1] - surface.value.box[0]))
const sy = computed(() => H / (surface.value.box[3] - surface.value.box[2]))
const toX = x => (x - surface.value.box[0]) * sx.value
const toY = y => H - (y - surface.value.box[2]) * sy.value

const grad = p => surface.value.g(p[0], p[1])
const gap = p => surface.value.f(p[0], p[1]) - surface.value.fmin

function heavyBall(step, beta) {
  let p = [...start.value], u = [0, 0]
  const out = [[...p]]
  for (let t = 0; t < steps.value; t++) {
    const g = grad(p)
    u = [beta * u[0] - step * g[0], beta * u[1] - step * g[1]]
    p = [p[0] + u[0], p[1] + u[1]]
    out.push([...p])
    if (!inside(p)) break
  }
  return out
}

function adam(step) {
  let p = [...start.value], m = [0, 0], v = [0, 0]
  const b1 = 0.9, b2 = 0.999, out = [[...p]]
  for (let t = 1; t <= steps.value; t++) {
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

function inside(p) {
  const [x0, x1, y0, y1] = surface.value.box
  const dx = x1 - x0, dy = y1 - y0
  return Number.isFinite(p[0]) && Number.isFinite(p[1]) &&
    p[0] > x0 - dx && p[0] < x1 + dx && p[1] > y0 - dy && p[1] < y1 + dy
}
function visible(path) {
  const out = []
  for (const p of path) { if (!inside(p)) break; out.push(p) }
  return out
}
const pts = path => visible(path).map(p => `${toX(p[0])},${toY(p[1])}`).join(' ')
const fmt = v => (!Number.isFinite(v) || v > 1e6 ? 'diverged' : v < 1e-4 ? v.toExponential(1) : v.toFixed(4))

// Contour background: log-spaced level bands, redrawn when the surface changes.
const background = ref('')
function drawBackground() {
  if (typeof document === 'undefined') return
  const s = surface.value
  const canvas = document.createElement('canvas')
  canvas.width = W; canvas.height = H
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(W, H)
  const band = new Int16Array(W * H)
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const x = s.box[0] + (i + 0.5) / sx.value
      const y = s.box[2] + (H - j - 0.5) / sy.value
      band[j * W + i] = Math.floor(3 * Math.log(1 + 4 * (s.f(x, y) - s.fmin)))
    }
  }
  for (let k = 0; k < W * H; k++) {
    const i = k % W, b = band[k]
    const edge = (i + 1 < W && band[k + 1] !== b) || (k + W < W * H && band[k + W] !== b)
    const shade = edge ? 190 : 255 - Math.min(b, 30) * 1.2
    img.data[4 * k] = shade; img.data[4 * k + 1] = shade; img.data[4 * k + 2] = shade; img.data[4 * k + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  background.value = canvas.toDataURL()
}
onMounted(drawBackground)
watch([surfaceId, kappa], drawBackground)

function onClick(e) {
  const r = e.currentTarget.getBoundingClientRect()
  const scale = W / r.width
  const x = surface.value.box[0] + (e.clientX - r.left) * scale / sx.value
  const y = surface.value.box[2] + (H - (e.clientY - r.top) * scale) / sy.value
  start.value = [x, y]
}
</script>

<style scoped>
.race { display: flex; gap: 1.6em; align-items: center; justify-content: center; margin-top: 2px; }
.race-left { display: flex; flex-direction: column; gap: 0.3em; width: 470px; }
.race-surfaces { display: flex; gap: 0.35em; font-size: 0.66em; }
.race-surfaces button { padding: 0.15em 0.6em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.race-surfaces button.on { background: #8c0000; color: #fff; }
.race-plot { border: 1px solid #9ba8aa; background: #fff; cursor: crosshair; }
.race-caption { font-size: 0.6em; line-height: 1.3; }
.muted { color: #667678; }
.race-controls { display: flex; flex-direction: column; gap: 0.32em; width: 330px; font-size: 0.68em; }
.race-controls label { display: grid; grid-template-columns: 1fr auto; gap: 0.1em 0.6em; }
.race-controls input[type=range] { grid-column: 1 / -1; width: 100%; accent-color: #8c0000; }
.race-limit { padding-left: 0.5em; border-left: 4px solid #617b67; font-weight: 600; }
.race-limit.bad { border-color: #8c0000; color: #8c0000; }
.race-table { border-collapse: collapse; margin-top: 0.3em; }
.race-table th { font-weight: 400; color: #667678; text-align: left; padding-right: 0.8em; }
.race-table td { padding: 0.08em 0.8em 0.08em 0; }
.mono { font-family: 'Fira Mono', monospace; }
</style>
