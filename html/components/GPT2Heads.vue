<!-- Written by Claude (2026-10-28): the 144 attention heads of GPT-2 (small)
     on a few sentences, for the Transformer demos. Data precomputed by
     scripts/make_gpt2_attention.py (html/data/gpt2_attention.json). The head
     grid is colored by three scores measured on a repeated random sequence. -->
<template>
  <div class="g2">
    <div class="g2-left">
      <select class="g2-sel" v-model.number="si">
        <option v-for="(s, i) in sentences" :key="i" :value="i">{{ s.text.replace(/\n\s*/g, ' ') }}</option>
      </select>
      <svg :width="12 * hc + 24" :height="12 * hc + 26" role="img" aria-label="Heads of GPT-2">
        <text v-for="h in 12" :key="'h' + h" :x="22 + (h - 1) * hc + hc / 2" :y="10" text-anchor="middle" class="ax">{{ h - 1 }}</text>
        <text v-for="l in 12" :key="'l' + l" :x="16" :y="16 + (l - 1) * hc + hc / 2 + 4" text-anchor="end" class="ax">{{ l - 1 }}</text>
        <g v-for="l in 12" :key="'r' + l">
          <rect v-for="h in 12" :key="h" :x="22 + (h - 1) * hc" :y="16 + (l - 1) * hc" :width="hc - 2" :height="hc - 2"
            :fill="headColor(l - 1, h - 1)" :stroke="sel[0] === l - 1 && sel[1] === h - 1 ? '#8c0000' : 'none'" stroke-width="2.5"
            style="cursor: pointer" @click="sel = [l - 1, h - 1]" />
        </g>
      </svg>
      <div class="legend">
        <span><i style="background:#6C8EBF"></i>previous token</span>
        <span><i style="background:#9ba8aa"></i>first token</span>
        <span><i style="background:#d79b00"></i>induction</span>
      </div>
      <div class="ax">rows: layers, columns: heads. Click a head.</div>
    </div>

    <div class="g2-right" v-if="tokens.length">
      <div class="g2-title">layer {{ sel[0] }}, head {{ sel[1] }}
        <span class="muted">— previous {{ pct('previous') }}, first {{ pct('first') }}, induction {{ pct('induction') }}</span></div>
      <svg :width="x0 + T * c + 50" :height="y0 + T * c + 4" role="img" aria-label="Attention matrix of the selected head">
        <text v-for="(t, j) in tokens" :key="'c' + j" :x="x0 + j * c + c / 2" :y="y0 - 5" class="tk"
          :transform="`rotate(-50 ${x0 + j * c + c / 2} ${y0 - 5})`">{{ vis(t) }}</text>
        <g v-for="i in T" :key="'r' + i" @mouseenter="hover = i - 1">
          <text :x="x0 - 6" :y="y0 + (i - 1) * c + c / 2 + 4" text-anchor="end" class="tk" :class="{ hl: hover === i - 1 }">{{ vis(tokens[i - 1]) }}</text>
          <rect v-for="j in T" :key="j" :x="x0 + (j - 1) * c" :y="y0 + (i - 1) * c" :width="c" :height="c"
            :fill="j > i ? '#fafafa' : shade(w(i - 1, j - 1))" stroke="#e1e5e6" stroke-width="0.5" />
        </g>
        <rect v-if="hover >= 0" :x="x0" :y="y0 + hover * c" :width="T * c" :height="c" fill="none" stroke="#8c0000" stroke-width="1.5" />
      </svg>
      <div class="g2-text">
        <span v-for="(t, j) in tokens" :key="j" class="wt" :class="{ q: j === hover }"
          :style="{ background: j <= hover ? shade(w(hover, j)) : 'transparent', color: j <= hover && w(hover, j) > 50 ? '#fff' : '#222' }">{{ vis(t) }}</span>
      </div>
    </div>
    <div v-else class="muted">{{ status }}</div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'

const props = defineProps({ initial: { type: Array, default: () => [4, 11] } })
const data = ref(null)
const status = ref('Loading GPT-2 attention…')
const si = ref(0)
const sel = ref(props.initial)
const hover = ref(-1)
const hc = 18, c = 16, x0 = 70, y0 = 54

const sentences = computed(() => (data.value ? data.value.sentences : []))
const tokens = computed(() => (data.value ? sentences.value[si.value].tokens : []))
const T = computed(() => tokens.value.length)
const flat = computed(() => sentences.value[si.value].attn[sel.value[0]][sel.value[1]])
const w = (i, j) => (i < 0 ? 0 : flat.value[(i * (i + 1)) / 2 + j])
const shade = v => `rgba(8, 69, 125, ${Math.min(1, 0.03 + (v / 100) * 1.2)})`
const vis = t => t.replace(/\n/g, '↵').replace(/ /g, '·')
const score = (k, l, h) => data.value.scores[k][l][h]
const pct = k => Math.round(100 * score(k, sel.value[0], sel.value[1])) + '%'

function headColor(l, h) {
  if (!data.value) return '#eee'
  const p = score('previous', l, h), f = score('first', l, h), ind = score('induction', l, h)
  const best = Math.max(p, ind)
  if (best > 0.4) return p >= ind ? `rgba(108,142,191,${0.3 + 0.7 * p})` : `rgba(215,155,0,${0.3 + 0.7 * ind})`
  return `rgba(155,168,170,${0.15 + 0.6 * f})`
}

onMounted(async () => {
  try {
    const res = await fetch(new URL('../../data/gpt2_attention.json', window.location.origin + import.meta.env.BASE_URL))
    data.value = await res.json()
    hover.value = Math.min(8, data.value.sentences[0].tokens.length - 1)
  } catch (err) {
    status.value = 'Could not load the GPT-2 attention data.'
  }
})
</script>

<style scoped>
.g2 { display: flex; gap: 1.6em; justify-content: center; align-items: flex-start; margin-top: 2px; }
.g2-left { display: flex; flex-direction: column; gap: 0.35em; font-size: 0.6em; width: 290px; }
.g2-sel { font-family: 'Fira Mono', monospace; font-size: 0.95em; padding: 0.1em; border: 1px solid #8c0000; color: #8c0000; max-width: 270px; }
.g2-sents button { padding: 0.1em 0.45em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; font-family: 'Fira Mono', monospace; font-size: 0.95em; }
.g2-sents button.on { background: #8c0000; color: #fff; }
.legend { display: flex; gap: 0.8em; flex-wrap: wrap; }
.legend i { display: inline-block; width: 0.9em; height: 0.9em; margin-right: 0.3em; vertical-align: -0.1em; }
.g2-right { display: flex; flex-direction: column; gap: 0.2em; font-size: 0.62em; }
.g2-title { font-weight: bold; }
.g2-text { max-width: 520px; line-height: 1.7; }
.wt { font-family: 'Fira Mono', monospace; padding: 1px 2px; margin-right: 1px; }
.wt.q { outline: 2px solid #8c0000; }
.tk { font-family: 'Fira Mono', monospace; font-size: 10px; fill: #333; }
.tk.hl { fill: #8c0000; font-weight: bold; }
.ax { font-size: 10px; fill: #667678; color: #667678; }
.muted { color: #667678; font-weight: normal; }
</style>
