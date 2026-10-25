<!-- Single- or multiple-answer question. Options may contain $...$ math.
     The explanation slot is shown after a correct answer or "Show solution";
     an optional aside slot (e.g. a figure) sits to the left of the options. -->
<template>
  <div class="quiz">
    <div :class="{ 'quiz-with-aside': $slots.aside }">
      <div v-if="$slots.aside" class="quiz-aside"><slot name="aside" /></div>
      <div class="quiz-options" :class="columns > 1 ? `cols-${columns}` : ''">
        <label
          v-for="(option, i) in options"
          :key="i"
          class="quiz-option"
          :class="optionClass(i)"
        >
          <input
            :type="multiple ? 'checkbox' : 'radio'"
            :name="id"
            :checked="selected.has(i)"
            :disabled="finished"
            @change="toggle(i)"
          />
          <span v-html="renderInline(option.text)" />
        </label>
      </div>
    </div>
    <div class="quiz-controls">
      <button class="primary" :disabled="finished || selected.size === 0" @click="check">Check</button>
      <button v-if="!finished" @click="reveal">Show solution</button>
      <span v-if="feedback" class="quiz-feedback" :class="feedback.kind">{{ feedback.text }}</span>
      <span v-else-if="multiple && !finished" class="muted">Select all that apply.</span>
    </div>
    <div v-if="finished" class="quiz-explanation"><slot name="explanation" /></div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { renderInline, recordAttempt, recordReveal } from './quiz.js'
import './quiz.css'

const props = defineProps({
  id: { type: String, required: true },
  // [{ text: 'option with $math$', correct: true }]
  options: { type: Array, required: true },
  multiple: { type: Boolean, default: false },
  columns: { type: Number, default: 1 },
})

const selected = ref(new Set())
const feedback = ref(null)
const state = ref('open') // open | correct | revealed
const finished = computed(() => state.value !== 'open')

function toggle(i) {
  const next = props.multiple ? new Set(selected.value) : new Set()
  if (props.multiple && next.has(i)) next.delete(i)
  else next.add(i)
  selected.value = next
  feedback.value = null
}

function mistakes() {
  return props.options.filter((o, i) => Boolean(o.correct) !== selected.value.has(i)).length
}

function check() {
  const errors = mistakes()
  recordAttempt(props.id, errors === 0)
  if (errors === 0) {
    state.value = 'correct'
    feedback.value = { kind: 'right', text: 'Correct.' }
  } else if (props.multiple) {
    feedback.value = { kind: 'wrong', text: `Not quite: ${errors} ${errors === 1 ? 'choice needs' : 'choices need'} changing.` }
  } else {
    feedback.value = { kind: 'wrong', text: 'Not quite. Try again.' }
  }
}

function reveal() {
  recordReveal(props.id)
  state.value = 'revealed'
  feedback.value = null
}

function optionClass(i) {
  const chosen = selected.value.has(i)
  const correct = Boolean(props.options[i].correct)
  if (!finished.value) return { selected: chosen }
  return { right: correct && chosen, wrong: chosen && !correct, missed: correct && !chosen }
}
</script>
