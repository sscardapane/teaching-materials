<!-- Progress table for one quiz, read from this browser's localStorage. -->
<template>
  <div class="quiz quiz-summary">
    <table>
      <tbody>
        <tr v-for="(item, i) in items" :key="item.id">
          <td class="muted">{{ i + 1 }}</td>
          <td><a href="#" @click.prevent="go(item.id)">{{ item.title }}</a></td>
          <td class="quiz-stars">{{ '★'.repeat(item.stars) }}</td>
          <td :class="statusOf(item.id).kind">{{ statusOf(item.id).text }}</td>
        </tr>
      </tbody>
    </table>
    <div class="quiz-controls">
      <span><strong>{{ solved }}</strong> of {{ items.length }} solved without the solution.</span>
      <button @click="reset">Reset my answers</button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'
import { progress, resetProgress } from './quiz.js'
import './quiz.css'

const props = defineProps({
  prefix: { type: String, required: true },
  // [{ id, title, stars, slide }] where slide is the question's slide number
  items: { type: Array, required: true },
})

const { $slidev } = useSlideContext()

function statusOf(id) {
  const e = progress[id]
  if (!e || (e.status === 'open' && e.attempts === 0)) return { kind: 'muted', text: 'not attempted' }
  if (e.status === 'correct') {
    return { kind: 'right', text: e.attempts === 1 ? '✓ first try' : `✓ after ${e.attempts} attempts` }
  }
  if (e.status === 'revealed') return { kind: 'wrong', text: 'solution shown' }
  return { kind: 'wrong', text: `open (${e.attempts} ${e.attempts === 1 ? 'attempt' : 'attempts'})` }
}

const solved = computed(() => props.items.filter((item) => progress[item.id]?.status === 'correct').length)

function go(id) {
  const item = props.items.find((i) => i.id === id)
  if (item?.slide) $slidev.nav.go(item.slide)
}

function reset() {
  resetProgress(props.prefix)
  // Question components keep their state until the page is reloaded.
  window.location.reload()
}
</script>

<style>
.quiz-summary table { width: 100%; border-collapse: collapse; font-size: .9em; }
.quiz-summary td { padding: 3px 10px; border-bottom: 1px solid #e1e6e6; }
.quiz-summary td.right { color: #2e7d4f; }
.quiz-summary td.wrong { color: var(--nnds-accent, #8c0000); }
.quiz-summary a { color: inherit; text-decoration: none; }
.quiz-summary a:hover { text-decoration: underline; }
</style>
