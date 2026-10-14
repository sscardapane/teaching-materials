<template>
  <div class="conv">
    <div class="conv-pane">
      <canvas ref="inputCanvas" :width="N" :height="N" class="conv-canvas" />
      <div class="conv-caption">input (one channel)</div>
    </div>

    <div class="conv-kernel">
      <div class="conv-grid">
        <input v-for="(w, i) in kernel" :key="i" type="text" inputmode="decimal"
          :value="show(w)" @change="e => setWeight(i, e.target.value)" />
      </div>
      <div class="conv-presets">
        <button v-for="(k, name) in presets" :key="name" :class="{ active: preset === name }" @click="choose(name)">{{ name }}</button>
      </div>
      <label class="conv-toggle"><input type="checkbox" v-model="absolute" /> show |output|</label>
      <label class="conv-toggle"><input type="checkbox" v-model="stride2" /> stride 2</label>
      <div class="conv-note">output {{ outSize }} × {{ outSize }}, same 9 weights at every pixel</div>
    </div>

    <div class="conv-pane">
      <canvas ref="outputCanvas" :width="N" :height="N" class="conv-canvas" />
      <div class="conv-caption">output</div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import imageUrl from '../public/astronaut_gray.png'

const N = 128
const presets = {
  identity: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  blur: [1, 1, 1, 1, 1, 1, 1, 1, 1].map(v => v / 9),
  'vertical edges': [-1, 0, 1, -2, 0, 2, -1, 0, 1],
  'horizontal edges': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
  sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
}
const preset = ref('vertical edges')
const kernel = reactive([...presets['vertical edges']])
const absolute = ref(true)
const stride2 = ref(false)
const inputCanvas = ref(null)
const outputCanvas = ref(null)
let pixels = null

const outSize = computed(() => (stride2.value ? N / 2 : N))

const show = w => String(Math.round(w * 100) / 100)
function setWeight(i, text) {
  const v = Number(String(text).replace(',', '.'))
  if (Number.isFinite(v)) kernel[i] = v
}

function choose(name) {
  preset.value = name
  presets[name].forEach((v, i) => { kernel[i] = v })
}

function render() {
  if (!pixels) return
  const r = stride2.value ? 2 : 1
  const M = N / r
  const out = new Float32Array(M * M)
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < M; j++) {
      let acc = 0
      for (let u = -1; u <= 1; u++) {
        for (let v = -1; v <= 1; v++) {
          const y = i * r + u, x = j * r + v
          if (y >= 0 && y < N && x >= 0 && x < N) acc += kernel[(u + 1) * 3 + (v + 1)] * pixels[y * N + x]
        }
      }
      out[i * M + j] = absolute.value ? Math.abs(acc) : acc
    }
  }
  let lo = Infinity, hi = -Infinity
  out.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v) })
  if (absolute.value) lo = 0
  const span = hi - lo || 1
  const ctx = outputCanvas.value.getContext('2d')
  const img = ctx.createImageData(N, N)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const v = out[Math.floor(y / r) * M + Math.floor(x / r)]
      let g = Math.round(255 * (v - lo) / span)
      if (absolute.value) g = 255 - g
      const k = (y * N + x) * 4
      img.data[k] = img.data[k + 1] = img.data[k + 2] = g
      img.data[k + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

onMounted(() => {
  const image = new Image()
  image.onload = () => {
    const ctx = inputCanvas.value.getContext('2d')
    ctx.drawImage(image, 0, 0, N, N)
    const data = ctx.getImageData(0, 0, N, N).data
    pixels = new Float32Array(N * N)
    for (let k = 0; k < N * N; k++) pixels[k] = data[k * 4] / 255
    render()
  }
  image.src = imageUrl
})

watch([kernel, absolute, stride2], render, { deep: true })
</script>

<style scoped>
.conv { display: flex; gap: 1.4em; align-items: center; justify-content: center; margin-top: 10px; }
.conv-pane { display: flex; flex-direction: column; align-items: center; gap: 0.3em; }
.conv-canvas { width: 256px; height: 256px; image-rendering: pixelated; border: 1px solid #9ba8aa; }
.conv-caption, .conv-note { color: #667678; font-size: 0.62em; }
.conv-kernel { display: flex; flex-direction: column; gap: 0.5em; width: 230px; font-size: 0.68em; }
.conv-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }
.conv-grid input { width: 100%; box-sizing: border-box; padding: 0.3em; border: 1px solid #6C8EBF; background: #eef3fa; text-align: center; font-family: 'Fira Mono', monospace; }
.conv-presets { display: flex; flex-wrap: wrap; gap: 0.3em; }
.conv-presets button { padding: 0.15em 0.5em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.conv-presets button.active { background: #8c0000; color: #fff; }
.conv-toggle { display: flex; gap: 0.4em; align-items: center; }
</style>
