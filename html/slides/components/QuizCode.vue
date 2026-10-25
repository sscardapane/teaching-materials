<!-- Coding question checked in the browser with Pyodide (NumPy only).
     `tests` runs after the student's code in the same namespace and should
     raise AssertionError with a readable message when something is wrong. -->
<template>
  <div class="quiz quiz-code">
    <textarea v-model="code" class="quiz-code-input" spellcheck="false" :rows="codeRows" @keydown.stop />
    <div class="quiz-controls">
      <button class="primary" :disabled="busy" @click="run">{{ busy ? busyLabel : 'Run tests' }}</button>
      <button v-if="!finished" :disabled="busy" @click="reveal">Show solution</button>
      <span v-if="feedback" class="quiz-feedback" :class="feedback.kind">{{ feedback.text }}</span>
      <span v-else-if="status === 'loading'" class="muted">First run downloads Python (about 10 MB).</span>
    </div>
    <pre v-if="output" class="quiz-code-output">{{ output }}</pre>
    <div v-if="finished" class="quiz-explanation">
      <template v-if="state === 'revealed'">
        <p><strong>One solution:</strong></p>
        <pre class="quiz-code-solution">{{ solution }}</pre>
      </template>
      <slot name="explanation" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { loadPython } from '../../components/pyodide.js'
import { recordAttempt, recordReveal } from './quiz.js'
import './quiz.css'

const props = defineProps({
  id: { type: String, required: true },
  initialCode: { type: String, required: true },
  tests: { type: String, required: true },
  solution: { type: String, required: true },
  codeRows: { type: Number, default: 8 },
})

const code = ref(props.initialCode)
const output = ref('')
const feedback = ref(null)
const status = ref('idle') // idle | loading | running
const state = ref('open')
const finished = computed(() => state.value !== 'open')
const busy = computed(() => status.value !== 'idle')
const busyLabel = computed(() => (status.value === 'loading' ? 'Loading Python…' : 'Running…'))

async function run() {
  feedback.value = null
  output.value = ''
  let py
  try {
    status.value = 'loading'
    py = await loadPython()
  } catch {
    status.value = 'idle'
    feedback.value = { kind: 'wrong', text: 'Could not load Python: check the network connection.' }
    return
  }
  status.value = 'running'
  let captured = ''
  py.setStdout({ batched: (s) => { captured += s + '\n' } })
  const namespace = py.globals.get('dict')()
  try {
    await py.runPythonAsync(code.value, { globals: namespace })
    await py.runPythonAsync(props.tests, { globals: namespace })
    recordAttempt(props.id, true)
    if (state.value === 'open') state.value = 'correct'
    feedback.value = { kind: 'right', text: 'All tests passed.' }
  } catch (err) {
    recordAttempt(props.id, false)
    feedback.value = { kind: 'wrong', text: 'Not yet: see the message below.' }
    captured += lastLine(String(err))
  } finally {
    namespace.destroy()
    output.value = captured.trim()
    status.value = 'idle'
  }
}

// Pyodide errors carry the full traceback; the last line is the useful part.
function lastLine(message) {
  const lines = message.trim().split('\n')
  return lines[lines.length - 1]
}

function reveal() {
  recordReveal(props.id)
  state.value = 'revealed'
  feedback.value = null
}
</script>

<style>
.quiz-code-input {
  width: 100%;
  box-sizing: border-box;
  padding: .7em;
  border: none;
  border-radius: 4px;
  background: #1e1e2e;
  color: #cdd6f4;
  font-family: 'Fira Mono', monospace;
  font-size: .82em;
  line-height: 1.4;
  resize: vertical;
}
.quiz-code-output, .quiz-code-solution {
  margin: 0 0 8px;
  padding: .5em .8em;
  border-left: 4px solid #8a9799;
  border-radius: 4px;
  background: #f5f5f5;
  font-family: 'Fira Mono', monospace;
  font-size: .78em;
  white-space: pre-wrap;
}
</style>
