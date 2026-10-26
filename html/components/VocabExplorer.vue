<template>
  <div class="vocab">
    <div class="vocab-left">
      <canvas ref="canvas" :width="W" :height="H" class="vocab-map"
        @mousemove="onMove" @mouseleave="hover = -1" @mousedown="onDown" @wheel.prevent="onWheel" @click="onClick" />
      <div class="vocab-tip mono" v-if="hover >= 0">{{ show(tokens[hover]) }} <span class="muted">#{{ hover }}</span></div>
      <div class="vocab-tip muted" v-else>{{ status }}</div>
    </div>

    <div class="vocab-right">
      <input v-model="query" class="vocab-search mono" placeholder="search the vocabulary, e.g. ing or ␣Rome" />
      <div class="vocab-cats">
        <button v-for="c in cats" :key="c.id" :class="{ on: cat === c.id && !query }" @click="pick(c.id)">{{ c.name }}</button>
      </div>
      <div class="muted vocab-note">{{ note }}</div>
      <div class="vocab-list">
        <span v-for="i in list.slice(0, 60)" :key="i" class="tok" @click="select(i)">{{ show(tokens[i]) }}</span>
      </div>
      <div v-if="selected >= 0" class="vocab-sel">
        <div><span class="tok big">{{ show(tokens[selected]) }}</span>
          <span class="muted">id {{ selected }}{{ selected >= 256 && selected < 50256 ? ', merge ' + (selected - 255) + ' of 50,000' : '' }}</span></div>
        <div class="muted">nearest neighbours in embedding space:</div>
        <div class="vocab-list"><span v-for="j in neighbours(selected)" :key="j" class="tok" @click="select(j)">{{ show(tokens[j]) }}</span></div>
      </div>
    </div>
  </div>
</template>

<script setup>
// Written by Claude (2026-10-26): GPT-2's 50,257 tokens, placed by a UMAP
// projection of their embeddings. The data is precomputed by
// scripts/make_vocab_map.py.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const W = 520, H = 360
const canvas = ref(null)
const tokens = ref([])
let X = [], Y = [], D = [], NB = [], K = 8
const status = ref('loading GPT-2 vocabulary…')
const hover = ref(-1), selected = ref(-1)
const query = ref(''), cat = ref('')
const view = { x0: 0, y0: 0, s: 1 }

const show = t => t === undefined ? '' : t
  .replace(/^ /, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥')
  .replace(/[\u0000-\u001f\u007f]/g, c => '\\x' + c.charCodeAt(0).toString(16).padStart(2, '0'))

const cats = [
  { id: 'long', name: 'longest', note: 'The longest tokens: strings so frequent in web text that BPE merged them whole.' },
  { id: 'last', name: 'last merges', note: 'The last merges BPE learned before stopping at 50,000: the rarest strings that still became tokens.' },
  { id: 'glitch', name: 'glitch tokens', note: 'Embeddings closest to the mean: tokens frequent in the tokenizer\'s data but almost never seen by the model in training (SolidGoldMagikarp and friends).' },
  { id: 'numbers', name: 'numbers', note: '' },
  { id: 'bytes', name: 'byte pieces', note: 'Single bytes that are only part of a character (shown as <XX>): accented letters, CJK and emoji are split into these when no merge covers them.' },
]

const isNum = t => /^ ?\d+$/.test(t)
const numberNote = computed(() => {
  if (!tokens.value.length) return ''
  const set = new Set(tokens.value)
  let bare = 0, spaced = 0
  for (let n = 0; n < 1000; n++) { if (set.has(String(n))) bare++; if (set.has(' ' + n)) spaced++ }
  return `Of the numbers 0–999, ${bare} are a single token on their own and ${spaced} with a leading space; the rest are split, e.g. ${missing(set)}.`
})
function missing(set) {
  const out = []
  for (let n = 100; n < 1000 && out.length < 4; n++) if (!set.has(String(n))) out.push(n)
  return out.join(', ')
}

const list = computed(() => {
  const T = tokens.value
  if (!T.length) return []
  const q = query.value.replace(/␣/g, ' ')
  if (q) return T.map((_, i) => i).filter(i => T[i].toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => T[a].length - T[b].length)
  const ids = T.map((_, i) => i)
  if (cat.value === 'long') return ids.sort((a, b) => T[b].length - T[a].length).slice(0, 300)
  if (cat.value === 'last') return ids.filter(i => i < 50256).reverse().slice(0, 300)
  if (cat.value === 'glitch') return ids.filter(i => i >= 256 && !T[i].includes('<')).sort((a, b) => D[a] - D[b]).slice(0, 120)
  if (cat.value === 'numbers') return ids.filter(i => isNum(T[i]))
  if (cat.value === 'bytes') return ids.filter(i => T[i].startsWith('<'))
  return []
})
const note = computed(() => {
  if (query.value) return `${list.value.length} tokens contain “${query.value}”.`
  if (cat.value === 'numbers') return numberNote.value
  const c = cats.find(c => c.id === cat.value)
  return c ? c.note : 'Hover the map, drag to pan, scroll to zoom, click a token to see its neighbours.'
})

function pick(id) { query.value = ''; cat.value = cat.value === id ? '' : id }
function select(i) { selected.value = i }
const neighbours = i => NB.slice(i * K, i * K + K)

const px = i => (X[i] - view.x0) * view.s * W / 1000
const py = i => H - (Y[i] - view.y0) * view.s * H / 1000

function draw() {
  const c = canvas.value
  if (!c || !X.length) return
  const ctx = c.getContext('2d')
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = 'rgba(108, 142, 191, 0.35)'
  for (let i = 0; i < X.length; i++) ctx.fillRect(px(i), py(i), 1.3, 1.3)
  const hl = list.value
  ctx.fillStyle = '#8c0000'
  for (const i of hl) ctx.fillRect(px(i) - 1.5, py(i) - 1.5, 3.5, 3.5)
  const zoomed = view.s >= 6
  if (zoomed) {
    ctx.fillStyle = '#333'; ctx.font = '10px Fira Mono, monospace'
    for (let i = 0; i < X.length; i++) {
      const x = px(i), y = py(i)
      if (x > 0 && x < W && y > 0 && y < H && (i % Math.max(1, Math.round(40 / view.s)) === 0 || hl.includes(i))) ctx.fillText(show(tokens.value[i]), x + 2, y - 2)
    }
  }
  for (const i of [selected.value, hover.value]) {
    if (i < 0) continue
    ctx.strokeStyle = '#d79b00'; ctx.lineWidth = 2
    ctx.beginPath(); ctx.arc(px(i), py(i), 5, 0, 7); ctx.stroke()
  }
  if (selected.value >= 0) {
    ctx.strokeStyle = 'rgba(215, 155, 0, 0.6)'; ctx.lineWidth = 1
    for (const j of neighbours(selected.value)) { ctx.beginPath(); ctx.moveTo(px(selected.value), py(selected.value)); ctx.lineTo(px(j), py(j)); ctx.stroke() }
  }
}
watch([list, hover, selected], draw)

function canvasXY(e) {
  const r = canvas.value.getBoundingClientRect()
  return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]
}
function nearest(mx, my) {
  let best = -1, bd = 64
  for (let i = 0; i < X.length; i++) {
    const dx = px(i) - mx, dy = py(i) - my, d = dx * dx + dy * dy
    if (d < bd) { bd = d; best = i }
  }
  return best
}
let drag = null, moved = false
function onDown(e) { drag = canvasXY(e); moved = false }
function onMove(e) {
  const [mx, my] = canvasXY(e)
  if (drag && e.buttons) {
    view.x0 -= (mx - drag[0]) * 1000 / (W * view.s)
    view.y0 += (my - drag[1]) * 1000 / (H * view.s)
    drag = [mx, my]; moved = true; draw(); return
  }
  hover.value = nearest(mx, my)
}
function onClick(e) {
  drag = null
  if (moved) return
  const [mx, my] = canvasXY(e)
  const i = nearest(mx, my)
  if (i >= 0) selected.value = i
}
function onWheel(e) {
  const [mx, my] = canvasXY(e)
  const fx = view.x0 + mx * 1000 / (W * view.s), fy = view.y0 + (H - my) * 1000 / (H * view.s)
  view.s = Math.min(80, Math.max(1, view.s * (e.deltaY < 0 ? 1.25 : 0.8)))
  view.x0 = fx - mx * 1000 / (W * view.s); view.y0 = fy - (H - my) * 1000 / (H * view.s)
  draw()
}
const up = () => { drag = null }

onMounted(async () => {
  window.addEventListener('mouseup', up)
  try {
    // The site keeps the data once, at <site>/data/, next to the nnds/ decks.
    const res = await fetch(new URL('../../data/gpt2_vocab_map.json', window.location.origin + import.meta.env.BASE_URL))
    const d = await res.json()
    X = d.x; Y = d.y; D = d.dist; NB = d.nbrs; K = d.k
    tokens.value = d.tokens
    status.value = `${d.tokens.length.toLocaleString('en')} GPT-2 tokens. Hover, drag, scroll to zoom.`
    draw()
  } catch (err) {
    status.value = 'Could not load the vocabulary map.'
  }
})
onBeforeUnmount(() => window.removeEventListener('mouseup', up))
</script>

<style scoped>
.vocab { display: flex; gap: 1.2em; align-items: flex-start; justify-content: center; margin-top: 2px; }
.vocab-left { display: flex; flex-direction: column; gap: 0.2em; }
.vocab-map { border: 1px solid #9ba8aa; background: #fff; cursor: grab; }
.vocab-tip { font-size: 0.62em; min-height: 1.3em; }
.vocab-right { display: flex; flex-direction: column; gap: 0.4em; width: 360px; font-size: 0.62em; }
.vocab-search { padding: 0.25em 0.5em; border: 1px solid #9ba8aa; }
.vocab-cats { display: flex; flex-wrap: wrap; gap: 0.3em; }
.vocab-cats button { padding: 0.12em 0.5em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.vocab-cats button.on { background: #8c0000; color: #fff; }
.vocab-note { line-height: 1.3; min-height: 2.6em; }
.vocab-list { display: flex; flex-wrap: wrap; gap: 3px; max-height: 8.5em; overflow-y: auto; }
.tok { padding: 0 0.35em; border: 1px solid #6C8EBF; background: #eef3fa; font-family: 'Fira Mono', monospace; cursor: pointer; white-space: pre; max-width: 330px; overflow: hidden; text-overflow: ellipsis; }
.tok.big { font-size: 1.3em; border-color: #d79b00; background: #fdf5e3; }
.vocab-sel { display: flex; flex-direction: column; gap: 0.3em; padding-top: 0.4em; border-top: 1px solid #9ba8aa; }
.muted { color: #667678; }
.mono { font-family: 'Fira Mono', monospace; }
</style>
