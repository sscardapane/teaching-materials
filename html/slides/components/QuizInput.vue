<!-- Free-answer question: a number (relative tolerance), a shape such as
     (32, 64), or an einsum specification (index names may differ). -->
<template>
  <div class="quiz">
    <div class="quiz-controls">
      <input
        v-model="answer"
        class="quiz-answer"
        :placeholder="placeholder || defaults[kind]"
        :disabled="finished"
        spellcheck="false"
        @keydown.enter.stop="check"
        @input="feedback = null"
      />
      <button class="primary" :disabled="finished || !answer.trim()" @click="check">Check</button>
      <button v-if="!finished" @click="reveal">Show solution</button>
      <span v-if="feedback" class="quiz-feedback" :class="feedback.kind">{{ feedback.text }}</span>
    </div>
    <div v-if="finished" class="quiz-explanation">
      <p v-if="state === 'revealed'"><strong>Answer:</strong> <span v-html="renderInline(solution)" /></p>
      <slot name="explanation" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { renderInline, recordAttempt, recordReveal, parseNumber, parseShape, canonicalEinsum } from './quiz.js'
import './quiz.css'

const props = defineProps({
  id: { type: String, required: true },
  kind: { type: String, default: 'number' }, // number | shape | einsum
  // number: a Number; shape: an Array of ints; einsum: a String such as 'btd,bsd->bts'
  expected: { type: [Number, Array, String], required: true },
  tolerance: { type: Number, default: 0 }, // relative, for numbers
  solution: { type: String, required: true }, // shown on "Show solution"; may contain $math$
  placeholder: { type: String, default: '' },
})

const defaults = { number: 'a number', shape: 'e.g. (2, 3, 4)', einsum: 'e.g. ij,jk->ik' }
const answer = ref('')
const feedback = ref(null)
const state = ref('open')
const finished = computed(() => state.value !== 'open')

function grade(text) {
  if (props.kind === 'number') {
    const value = parseNumber(text)
    if (value === null) return { valid: false, hint: 'Please enter a number.' }
    const scale = Math.max(Math.abs(props.expected), 1e-12)
    return { valid: true, correct: Math.abs(value - props.expected) <= props.tolerance * scale + 1e-12 }
  }
  if (props.kind === 'shape') {
    const dims = parseShape(text)
    if (!dims) return { valid: false, hint: 'Write the shape as a list of integers, e.g. (2, 3).' }
    return { valid: true, correct: dims.join(',') === props.expected.join(',') }
  }
  const spec = canonicalEinsum(text)
  if (!spec) return { valid: false, hint: 'Write a full specification with an explicit output, e.g. ij,jk->ik.' }
  return { valid: true, correct: spec === canonicalEinsum(props.expected) }
}

function check() {
  if (finished.value || !answer.value.trim()) return
  const result = grade(answer.value)
  if (!result.valid) {
    feedback.value = { kind: 'wrong', text: result.hint }
    return
  }
  recordAttempt(props.id, result.correct)
  if (result.correct) {
    state.value = 'correct'
    feedback.value = { kind: 'right', text: 'Correct.' }
  } else {
    feedback.value = { kind: 'wrong', text: 'Not quite. Try again.' }
  }
}

function reveal() {
  recordReveal(props.id)
  state.value = 'revealed'
  feedback.value = null
}
</script>
