<template>
  <div class="rf">
    <div class="rf-controls">
      <label>layers <b>{{ L }}</b><input type="range" min="1" max="5" v-model.number="L" /></label>
      <label>kernel size <b>{{ s }}</b><input type="range" min="2" max="5" v-model.number="s" /></label>
      <div class="rf-modes">
        <button v-for="m in modes" :key="m.id" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <label class="rf-check"><input type="checkbox" v-model="causal" /> causal</label>
    </div>

    <svg :width="W" :height="Hsvg" role="img" aria-label="Receptive field of the top output">
      <g v-for="l in L + 1" :key="'layer' + l">
        <text :x="4" :y="rowY(l - 1) + 5" :style="{ fontSize: '13px' }" fill="#667678">{{ l === 1 ? 'input' : 'layer ' + (l - 1) }}</text>
      </g>
      <line v-for="(e, i) in edges" :key="'e' + i" :x1="toX(e[1])" :y1="rowY(e[0] - 1)" :x2="toX(e[2])" :y2="rowY(e[0])"
        stroke="#6C8EBF" stroke-width="1.4" />
      <g v-for="l in L + 1" :key="'u' + l">
        <circle v-for="c in unitsOf(l - 1)" :key="c" :cx="toX(c)" :cy="rowY(l - 1)" r="6"
          :fill="active[l - 1].has(c) ? '#DAE8FC' : '#fff'" :stroke="active[l - 1].has(c) ? '#6C8EBF' : '#c4c4c4'" stroke-width="1.4" />
      </g>
    </svg>

    <div class="rf-readout">
      receptive field of the top output: <b>{{ traced.clipped ? '> ' + R : R }}</b> input positions
      <span class="rf-formula">{{ formula }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const W = 830, span = 22, left = 70
const L = ref(3)
const s = ref(3)
const mode = ref('plain')
const causal = ref(false)
const modes = [
  { id: 'plain', label: 'plain' },
  { id: 'stride', label: 'stride 2 in layer 1' },
  { id: 'dilated', label: 'dilation 1, 2, 4, …' },
]

const Hsvg = computed(() => 40 + 52 * L.value)
const rowY = l => Hsvg.value - 20 - 52 * l
const toX = c => left + (c + span) * ((W - left - 10) / (2 * span))

// Cumulative stride before layer l and dilation of layer l (1-based layers).
const strideUpTo = l => (mode.value === 'stride' && l >= 1 ? 2 : 1)
const dilation = l => (mode.value === 'dilated' ? 2 ** (l - 1) : 1)
const offsets = computed(() => {
  const k = s.value - 1
  if (causal.value) return Array.from({ length: s.value }, (_, u) => u - k)
  const lo = -Math.floor(k / 2)
  return Array.from({ length: s.value }, (_, u) => lo + u)
})

function unitsOf(l) {
  const step = strideUpTo(l)
  const out = []
  for (let c = -span; c <= span; c++) if (c % step === 0) out.push(c)
  return out
}

const traced = computed(() => {
  const act = Array.from({ length: L.value + 1 }, () => new Set())
  const edges = []
  let clipped = false
  const top = causal.value ? 2 * Math.floor(span / 4) : 0
  act[L.value].add(top)
  for (let l = L.value; l >= 1; l--) {
    const below = strideUpTo(l - 1)
    for (const c of act[l]) {
      for (const u of offsets.value) {
        const src = c + u * dilation(l) * below
        if (src < -span || src > span) { clipped = true; continue }
        act[l - 1].add(src)
        edges.push([l, src, c])
      }
    }
  }
  return { act, edges, clipped }
})
const active = computed(() => traced.value.act)
const edges = computed(() => traced.value.edges)
const R = computed(() => {
  const xs = [...active.value[0]]
  return Math.max(...xs) - Math.min(...xs) + 1
})
const formula = computed(() => {
  const k = s.value - 1
  if (mode.value === 'plain') return `= 1 + L(s−1) = 1 + ${L.value}·${k}`
  if (mode.value === 'stride') return `= 1 + (s−1) + 2(L−1)(s−1): later layers grow twice as fast`
  return `= 1 + (s−1)(2^L − 1): exponential in depth`
})
</script>

<style scoped>
.rf { display: flex; flex-direction: column; align-items: center; gap: 0.4em; margin-top: 2px; }
.rf-controls { display: flex; gap: 1.4em; align-items: center; font-size: 0.66em; }
.rf-controls label { display: flex; flex-direction: column; min-width: 120px; }
.rf-controls input[type=range] { accent-color: #8c0000; }
.rf-check { flex-direction: row !important; gap: 0.4em; align-items: center; }
.rf-modes { display: flex; gap: 0.3em; }
.rf-modes button { padding: 0.15em 0.55em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.rf-modes button.active { background: #8c0000; color: #fff; }
.rf-readout { font-size: 0.72em; }
.rf-formula { margin-left: 0.6em; color: #667678; font-family: 'Fira Mono', monospace; font-size: 0.9em; }
</style>
