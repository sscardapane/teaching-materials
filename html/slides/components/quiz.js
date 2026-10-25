// Shared state and helpers for the self-evaluation quiz components.
// Progress lives only in the student's browser (localStorage); nothing is sent anywhere.
import { reactive } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const KEY = 'nnds-quiz-v1'

function load() {
  try { return JSON.parse(window.localStorage.getItem(KEY)) || {} } catch { return {} }
}

function save() {
  try { window.localStorage.setItem(KEY, JSON.stringify(progress)) } catch { /* private mode */ }
}

// progress[id] = { attempts, status: 'open' | 'correct' | 'revealed' }
export const progress = reactive(typeof window === 'undefined' ? {} : load())

function entry(id) {
  return progress[id] || { attempts: 0, status: 'open' }
}

export function recordAttempt(id, correct) {
  const e = { ...entry(id) }
  if (e.status === 'open') {
    e.attempts += 1
    if (correct) e.status = 'correct'
  }
  progress[id] = e
  save()
}

export function recordReveal(id) {
  const e = { ...entry(id) }
  if (e.status === 'open') e.status = 'revealed'
  progress[id] = e
  save()
}

export function resetProgress(prefix) {
  for (const id of Object.keys(progress)) {
    if (id.startsWith(prefix)) delete progress[id]
  }
  save()
}

// Render text with inline $...$ math. Option texts are written by the instructor,
// so plain HTML (for example <code>) is allowed.
export function renderInline(text) {
  return text.split(/(\$[^$]+\$)/g).map((part) => (
    part.length > 2 && part.startsWith('$') && part.endsWith('$')
      ? katex.renderToString(part.slice(1, -1), { throwOnError: false, strict: false })
      : part
  )).join('')
}

// --- answer parsers for QuizInput -------------------------------------------

export function parseNumber(text) {
  const cleaned = text.replace(/[\s_,]/g, '')
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(cleaned)) return null
  return Number(cleaned)
}

export function parseShape(text) {
  const dims = text.match(/\d+/g)
  return dims ? dims.map(Number) : null
}

// Rename einsum indices by order of first appearance, so that
// "btd,bsd->bts" and "ijk,ilk->ijl" compare equal.
export function canonicalEinsum(text) {
  const spec = text.replace(/\s|["']/g, '')
  if (!/^[a-zA-Z]+(,[a-zA-Z]+)*->[a-zA-Z]*$/.test(spec)) return null
  const names = new Map()
  return spec.replace(/[a-zA-Z]/g, (c) => {
    if (!names.has(c)) names.set(c, String.fromCharCode(97 + names.size))
    return names.get(c)
  })
}
