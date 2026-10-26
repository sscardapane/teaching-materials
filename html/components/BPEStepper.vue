<template>
  <div class="bpe">
    <div class="bpe-left">
      <table class="bpe-words">
        <tr><th>word</th><th>count</th><th>current tokens</th></tr>
        <tr v-for="w in words" :key="w.word">
          <td class="mono">{{ w.word }}</td>
          <td>×{{ w.count }}</td>
          <td><span v-for="(t, i) in w.tokens" :key="i" :class="['tok', isNew(t) ? 'fresh' : '', nextPair(w.tokens, i) ? 'next' : '']">{{ t }}</span></td>
        </tr>
      </table>
      <div class="bpe-stats">
        vocabulary: <b>{{ vocab.length }}</b> tokens ·
        corpus length: <b>{{ corpusLength }}</b> tokens
      </div>
    </div>

    <div class="bpe-right">
      <div class="bpe-head">most frequent adjacent pairs (counted with word frequencies)</div>
      <table class="bpe-pairs">
        <tr v-for="(p, i) in topPairs" :key="p.key" :class="{ best: i === 0 }">
          <td class="mono">({{ p.a }}, {{ p.b }})</td><td>{{ p.n }}</td>
        </tr>
      </table>
      <div class="bpe-buttons">
        <button @click="merge" :disabled="!topPairs.length">Merge top pair</button>
        <button @click="reset">Reset</button>
      </div>
      <div class="bpe-rules">
        merge rules, in order:
        <span v-for="(r, i) in rules" :key="i" class="mono">{{ i + 1 }}. {{ r }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

// The toy corpus of the CNN slides (hug, hugs, pug), extended with pun and bun.
const corpus = [['hug', 5], ['hugs', 3], ['pug', 2], ['pun', 4], ['bun', 2]]
const words = ref([])
const rules = ref([])
const lastMerged = ref('')

function reset() {
  words.value = corpus.map(([word, count]) => ({ word, count, tokens: word.split('') }))
  rules.value = []
  lastMerged.value = ''
}
reset()

const vocab = computed(() => {
  const base = new Set(corpus.flatMap(([w]) => w.split('')))
  rules.value.forEach(r => base.add(r.split(' → ')[1]))
  return [...base]
})
const corpusLength = computed(() => words.value.reduce((a, w) => a + w.count * w.tokens.length, 0))
const topPairs = computed(() => {
  const counts = new Map()
  for (const w of words.value) {
    for (let i = 0; i + 1 < w.tokens.length; i++) {
      const key = w.tokens[i] + '\u0000' + w.tokens[i + 1]
      counts.set(key, (counts.get(key) || 0) + w.count)
    }
  }
  return [...counts.entries()]
    .map(([key, n]) => { const [a, b] = key.split('\u0000'); return { key, a, b, n } })
    .sort((x, y) => y.n - x.n || (x.a + x.b).localeCompare(y.a + y.b))
    .slice(0, 5)
})
const isNew = t => t === lastMerged.value
// Tokens that the next merge will join, so students can check the count.
function nextPair(tokens, i) {
  const best = topPairs.value[0]
  if (!best) return false
  const at = j => j >= 0 && j + 1 < tokens.length && tokens[j] === best.a && tokens[j + 1] === best.b
  return at(i) || at(i - 1)
}

function merge() {
  const best = topPairs.value[0]
  if (!best) return
  const joined = best.a + best.b
  words.value = words.value.map(w => {
    const out = []
    for (let i = 0; i < w.tokens.length; i++) {
      if (i + 1 < w.tokens.length && w.tokens[i] === best.a && w.tokens[i + 1] === best.b) { out.push(joined); i++ }
      else out.push(w.tokens[i])
    }
    return { ...w, tokens: out }
  })
  rules.value = [...rules.value, `${best.a} + ${best.b} → ${joined}`]
  lastMerged.value = joined
}
</script>

<style scoped>
.bpe { display: flex; gap: 2.2em; justify-content: center; align-items: flex-start; margin-top: 8px; font-size: 0.7em; }
.mono { font-family: 'Fira Mono', monospace; }
.bpe-words { border-collapse: collapse; }
.bpe-words th, .bpe-head { color: #667678; font-weight: 400; text-align: left; padding-right: 1em; }
.bpe-words td { padding: 0.2em 1em 0.2em 0; }
.tok { display: inline-block; margin-right: 3px; padding: 0 0.35em; border: 1px solid #6C8EBF; background: #eef3fa; font-family: 'Fira Mono', monospace; }
.tok.fresh { border-color: #B85450; background: #f8e9e8; }
.tok.next { outline: 2px solid #d79b00; outline-offset: 1px; }
.bpe-stats { margin-top: 0.7em; }
.bpe-right { display: flex; flex-direction: column; gap: 0.5em; width: 300px; }
.bpe-pairs td { padding: 0.1em 1.2em 0.1em 0; }
.bpe-pairs tr.best td { color: #8c0000; font-weight: 600; }
.bpe-buttons { display: flex; gap: 0.4em; }
.bpe-buttons button { padding: 0.2em 0.6em; border: 1px solid #8c0000; background: #fff; color: #8c0000; cursor: pointer; }
.bpe-buttons button:disabled { opacity: 0.4; cursor: default; }
.bpe-rules { display: flex; flex-direction: column; color: #667678; }
</style>
