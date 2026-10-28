<!-- Written by Claude (2026-10-28): a small RASP interpreter for the Transformer
     demos (Weiss et al. 2021, "Thinking like transformers"). Every select is an
     attention pattern (rows: queries, columns: keys), every aggregate reads
     values through it, and elementwise maps play the role of the MLPs.
     Aggregation of categorical values: one selected position gives its value;
     several give their value if they all agree, "?" otherwise; none gives "·". -->
<template>
  <div class="rasp">
    <div class="rasp-left">
      <div class="rasp-progs">
        <button v-for="p in programs" :key="p.id" :class="{ on: prog === p.id }" @click="choose(p.id)">{{ p.label }}</button>
      </div>
      <label class="rasp-input">input
        <input v-model="text" maxlength="12" spellcheck="false" />
      </label>
      <div class="rasp-code">
        <div v-for="(s, i) in steps" :key="i" class="line" :class="{ on: i === cur }" @click="cur = i">
          <span class="tag" :class="s.kind === 'map' ? 'mlp' : 'att'">{{ s.kind === 'map' ? 'MLP' : 'attn' }}</span>
          <code>{{ s.code }}</code>
        </div>
      </div>
      <div class="muted">{{ current.note || note }}</div>
    </div>

    <div class="rasp-right">
      <svg v-if="current.matrix" :width="mx0 + n * mc + 8" :height="my0 + n * mc + 6" role="img" aria-label="Selector">
        <text :x="mx0 + (n * mc) / 2" :y="11" text-anchor="middle" class="ax">keys: {{ current.keysName }}</text>
        <text :x="10" :y="my0 + (n * mc) / 2" text-anchor="middle" class="ax"
          :transform="`rotate(-90 10 ${my0 + (n * mc) / 2})`">queries: {{ current.queriesName }}</text>
        <text v-for="(v, j) in current.keys" :key="'k' + j" :x="mx0 + j * mc + mc / 2" :y="my0 - 6" text-anchor="middle" class="tk">{{ show(v) }}</text>
        <text v-for="(v, i) in current.queries" :key="'q' + i" :x="mx0 - 6" :y="my0 + i * mc + mc / 2 + 4" text-anchor="end" class="tk">{{ show(v) }}</text>
        <g v-for="(row, i) in current.matrix" :key="i">
          <rect v-for="(b, j) in row" :key="j" :x="mx0 + j * mc" :y="my0 + i * mc" :width="mc" :height="mc"
            :fill="b ? '#08457d' : '#fff'" stroke="#c9d1d3" stroke-width="0.6" />
        </g>
      </svg>
      <div v-else class="rasp-nomatrix muted">An elementwise operation: each position is updated on its own, like an MLP. No attention pattern.</div>

      <table class="rasp-seqs">
        <tr v-for="(row, k) in shownSeqs" :key="row.name" :class="{ hl: row.name === current.out }">
          <th>{{ row.name }}</th>
          <td v-for="(v, j) in row.values" :key="j">{{ show(v) }}</td>
        </tr>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({ initial: { type: String, default: 'reverse' } })
const text = ref('hello')
const prog = ref(props.initial)
const cur = ref(0)
const note = 'Click a line to see its attention pattern. Edit the input to rerun the program.'

// --- RASP primitives -------------------------------------------------------
const select = (keys, queries, pred) => queries.map((q, i) => keys.map((k, j) => !!pred(k, q, j, i)))
function aggregate(sel, values, fallback = '·') {
  return sel.map(row => {
    const picked = values.filter((_, j) => row[j])
    if (!picked.length) return fallback
    if (typeof picked[0] === 'number' && picked.some(v => v !== picked[0])) return +(picked.reduce((a, b) => a + b, 0) / picked.length).toFixed(2)
    return picked.every(v => v === picked[0]) ? picked[0] : '?'
  })
}
const width = sel => sel.map(row => row.filter(Boolean).length)

// Each program is a list of steps; a step reads the environment and adds one sequence.
const S = (code, out, keysName, queriesName, pred, note) => ({
  kind: 'select', code, out, keysName, queriesName, note,
  run: env => { const m = select(env[keysName], env[queriesName], pred); env[out] = m; return { matrix: m, keys: env[keysName], queries: env[queriesName] } },
})
const A = (code, out, selName, valName, note) => ({
  kind: 'aggregate', code, out, note,
  run: env => { env[out] = aggregate(env[selName], env[valName]); return sel(env, selName) },
})
const Wd = (code, out, selName, note) => ({
  kind: 'width', code, out, note,
  run: env => { env[out] = width(env[selName]); return sel(env, selName) },
})
const M = (code, out, f, note) => ({ kind: 'map', code, out, note, run: env => { env[out] = f(env); return {} } })
const sel = (env, name) => ({ matrix: env[name], keys: env[name + ':k'], queries: env[name + ':q'] })

const lengthSteps = [
  S('all = select(tokens, tokens, True)', 'all', 'tokens', 'tokens', () => true,
    'Every query selects every key: a uniform attention pattern.'),
  Wd('length = selector_width(all)', 'length', 'all',
    'How many keys each query selects. Attention can compute this with a uniform pattern (plus an MLP).'),
]
const programs = [
  { id: 'length', label: 'length', steps: lengthSteps },
  {
    id: 'reverse', label: 'reverse', steps: [
      ...lengthSteps,
      M('opp = length - indices - 1', 'opp', e => e.length.map((L, i) => L - i - 1), 'Elementwise arithmetic: an MLP at each position.'),
      S('flip = select(indices, opp, ==)', 'flip', 'indices', 'opp', (k, q) => k === q,
        'Query i looks for the key whose index equals opp[i]: an anti-diagonal pattern.'),
      A('reverse = aggregate(flip, tokens)', 'reverse', 'flip', 'tokens', 'Each position copies the token it selected.'),
    ],
  },
  {
    id: 'hist', label: 'histogram', steps: [
      S('same = select(tokens, tokens, ==)', 'same', 'tokens', 'tokens', (k, q) => k === q,
        'Each token selects all positions holding the same token.'),
      Wd('hist = selector_width(same)', 'hist', 'same', 'Counting the selected keys gives the number of occurrences.'),
    ],
  },
  {
    id: 'sort', label: 'sort', steps: [
      S('smaller = select(tokens, tokens, <)', 'smaller', 'tokens', 'tokens', (k, q, j, i) => k < q || (k === q && j < i),
        'Key j is selected if its token comes before the query token (ties broken by position).'),
      Wd('target = selector_width(smaller)', 'target', 'smaller', 'The number of smaller tokens is the position of each token after sorting.'),
      S('move = select(target, indices, ==)', 'move', 'target', 'indices', (k, q) => k === q,
        'Output position i looks for the token whose target position is i.'),
      A('sorted = aggregate(move, tokens)', 'sorted', 'move', 'tokens'),
    ],
  },
  {
    id: 'induction', label: 'induction', text: 'abcdabcab', steps: [
      S('prev = select(indices, indices, k == q - 1)', 'prev', 'indices', 'indices', (k, q) => k === q - 1,
        'Layer 1: a previous-token head, like head 11 of layer 4 in GPT-2.'),
      A('prev_token = aggregate(prev, tokens)', 'prev_token', 'prev', 'tokens',
        'Each position now also stores the token before it, in its residual stream.'),
      S('match = select(prev_token, tokens, ==)', 'match', 'prev_token', 'tokens', (k, q, j, i) => k === q && j <= i,
        'Layer 2 (causal): position t selects the positions whose previous token equals token t.'),
      A('next = aggregate(match, tokens)', 'next', 'match', 'tokens',
        'It copies what followed the earlier occurrences: the prediction of an induction head.'),
    ],
  },
]

function choose(id) {
  prog.value = id
  cur.value = 0
  const p = programs.find(q => q.id === id)
  if (p.text) text.value = p.text
  else if (text.value === 'abcdabcab') text.value = 'hello'
}

const steps = computed(() => programs.find(p => p.id === prog.value).steps)
const run = computed(() => {
  const toks = [...text.value]
  const env = { tokens: toks, indices: toks.map((_, i) => i) }
  const order = ['tokens', 'indices']
  const views = steps.value.map(s => {
    const v = s.run(env)
    if (s.kind === 'select') { env[s.out + ':k'] = v.keys; env[s.out + ':q'] = v.queries }
    else order.push(s.out)
    return { ...v, out: s.out, note: s.note, keysName: s.keysName, queriesName: s.queriesName }
  })
  if (prog.value === 'induction') { env.actual_next = toks.map((_, i) => toks[i + 1] ?? '·'); order.push('actual_next') }
  return { env, order, views }
})
const n = computed(() => text.value.length)
const current = computed(() => {
  const v = run.value.views[Math.min(cur.value, run.value.views.length - 1)]
  const s = steps.value[Math.min(cur.value, steps.value.length - 1)]
  const selName = s.kind === 'select' ? s.out : null
  return { ...v, keysName: v.keysName ?? keysOf(s), queriesName: v.queriesName ?? queriesOf(s), selName }
})
const keysOf = s => { const st = steps.value.find(x => x.kind === 'select' && s.code.includes(x.out)); return st ? st.keysName : '' }
const queriesOf = s => { const st = steps.value.find(x => x.kind === 'select' && s.code.includes(x.out)); return st ? st.queriesName : '' }
const shownSeqs = computed(() => {
  const upto = new Set(['tokens', 'indices', 'actual_next', ...steps.value.slice(0, cur.value + 1).map(s => s.out)])
  return run.value.order.filter(k => upto.has(k)).map(k => ({ name: k, values: run.value.env[k] }))
})
const show = v => (v === ' ' ? '␣' : v)
const mc = 20, mx0 = 50, my0 = 32
</script>

<style scoped>
.rasp { display: flex; gap: 1.8em; justify-content: center; align-items: flex-start; margin-top: 2px; font-size: 0.6em; }
.rasp-left { display: flex; flex-direction: column; gap: 0.5em; width: 440px; }
.rasp-progs { display: flex; flex-wrap: wrap; gap: 0.3em; }
.rasp-progs button { padding: 0.12em 0.5em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.rasp-progs button.on { background: #8c0000; color: #fff; }
.rasp-input { display: flex; gap: 0.6em; align-items: center; }
.rasp-input input { font-family: 'Fira Mono', monospace; font-size: 1.15em; padding: 0.1em 0.4em; border: 1px solid #9ba8aa; width: 11em; }
.rasp-code { display: flex; flex-direction: column; border: 1px solid #c9d1d3; }
.line { display: flex; gap: 0.5em; align-items: center; padding: 0.22em 0.5em; cursor: pointer; }
.line.on { background: #fdf0f0; outline: 1px solid #8c0000; }
.line code { font-family: 'Fira Mono', monospace; font-size: 0.98em; white-space: pre; }
.tag { font-size: 0.8em; padding: 0 0.35em; border-radius: 2px; color: #fff; min-width: 2.6em; text-align: center; }
.tag.att { background: #6C8EBF; } .tag.mlp { background: #82B366; }
.rasp-right { display: flex; flex-direction: column; gap: 0.4em; align-items: flex-start; }
.rasp-nomatrix { width: 300px; padding: 2em 0; }
.rasp-seqs { border-collapse: collapse; font-family: 'Fira Mono', monospace; font-size: 0.9em; }
.rasp-seqs th { text-align: right; padding: 0.1em 0.6em 0.1em 0; font-weight: normal; color: #667678; }
.rasp-seqs td { border: 1px solid #c9d1d3; min-width: 1.5em; text-align: center; padding: 0 0.15em; }
.rasp-seqs tr.hl td { background: #eaf1fa; border-color: #6C8EBF; }
.rasp-seqs tr.hl th { color: #8c0000; }
.tk { font-family: 'Fira Mono', monospace; font-size: 12px; fill: #333; }
.ax { font-size: 10px; fill: #667678; }
.muted { color: #667678; }
</style>
