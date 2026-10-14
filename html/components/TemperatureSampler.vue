<template>
  <div class="temp">
    <div class="temp-left">
      <div class="temp-prompt"><span class="mono">Check this</span> → next token?</div>
      <svg :width="W" :height="H" role="img" aria-label="Next-token probabilities">
        <g v-for="(tok, i) in vocab" :key="tok">
          <text :x="0" :y="i * row + 17" :style="{ fontSize: '15px' }" class="mono">{{ tok }}</text>
          <rect :x="labelW" :y="i * row + 4" :width="probs[i] * (W - labelW - 54)" :height="row - 9"
            :fill="i === 0 ? '#B85450' : '#6C8EBF'" />
          <text :x="labelW + probs[i] * (W - labelW - 54) + 6" :y="i * row + 17" :style="{ fontSize: '13px' }" fill="#667678">{{ (100 * probs[i]).toFixed(0) }}%</text>
        </g>
      </svg>
    </div>

    <div class="temp-controls">
      <label>Temperature τ <b>{{ tau.toFixed(2) }}</b>
        <input type="range" min="0.05" max="3" step="0.05" v-model.number="tau" /></label>
      <div class="temp-presets">
        <button @click="tau = 0.1">τ → 0 (greedy)</button>
        <button @click="tau = 1">τ = 1</button>
        <button @click="tau = 2.5">τ = 2.5</button>
      </div>
      <div class="temp-stat">entropy: <b>{{ entropy.toFixed(2) }}</b> bits (max {{ Math.log2(vocab.length).toFixed(2) }})</div>
      <button class="temp-sample" @click="draw">Sample 12 continuations</button>
      <div class="temp-chips">
        <span v-for="(t, i) in samples" :key="i" :class="['chip', t === vocab[0] ? 'top' : '']">{{ t }}</span>
      </div>
      <div class="temp-note">p = softmax(logits / τ), as in the linear-models lecture</div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const vocab = ['out', 'now', 'again', 'video', 'one', '!', 'idea', 'banana']
const logits = [3.0, 2.0, 1.6, 1.2, 0.9, 0.6, 0.1, -1.5]
const W = 420, row = 30, labelW = 70
const H = vocab.length * row

const tau = ref(1)
const probs = computed(() => {
  const z = logits.map(l => l / tau.value)
  const m = Math.max(...z)
  const e = z.map(v => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map(v => v / s)
})
const entropy = computed(() => -probs.value.reduce((a, p) => a + (p > 0 ? p * Math.log2(p) : 0), 0))

// Small seeded generator, so a click gives a fresh but reproducible-looking draw.
let seed = 7
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
const samples = ref([])
function draw() {
  samples.value = Array.from({ length: 12 }, () => {
    let r = rand(), i = 0
    while (i < probs.value.length - 1 && r > probs.value[i]) { r -= probs.value[i]; i++ }
    return vocab[i]
  })
}
watch(tau, draw)
draw()
</script>

<style scoped>
.temp { display: flex; gap: 2em; align-items: center; justify-content: center; margin-top: 8px; }
.temp-prompt { margin-bottom: 0.4em; font-size: 0.75em; }
.mono { font-family: 'Fira Mono', monospace; }
.temp-controls { display: flex; flex-direction: column; gap: 0.5em; width: 340px; font-size: 0.68em; }
.temp-controls label { display: grid; grid-template-columns: 1fr auto; }
.temp-controls input[type=range] { grid-column: 1 / -1; width: 100%; accent-color: #8c0000; }
.temp-presets { display: flex; gap: 0.3em; }
.temp-presets button, .temp-sample { padding: 0.18em 0.55em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.temp-chips { display: flex; flex-wrap: wrap; gap: 0.3em; min-height: 3.6em; }
.chip { padding: 0.05em 0.45em; border: 1px solid #6C8EBF; background: #eef3fa; font-family: 'Fira Mono', monospace; }
.chip.top { border-color: #B85450; background: #f8e9e8; }
.temp-note { color: #667678; font-size: 0.9em; }
</style>
