<!-- Before/after picture of a linear map applied to the rows of X (H = XW).
     The shape is an "F" (it has no symmetry, so reflections and rotations
     are distinguishable); one marked point helps to read off the map. -->
<template>
  <div class="linear-map-figure">
    <figure v-for="panel in panels" :key="panel.label">
      <svg :viewBox="`${-R} ${-R} ${2 * R} ${2 * R}`" role="img" :aria-label="panel.label">
        <g class="grid">
          <line v-for="k in ticks" :key="`v${k}`" :x1="k" :y1="-R" :x2="k" :y2="R" />
          <line v-for="k in ticks" :key="`h${k}`" :x1="-R" :y1="k" :x2="R" :y2="k" />
        </g>
        <line class="axis" :x1="-R" y1="0" :x2="R" y2="0" />
        <line class="axis" x1="0" :y1="-R" x2="0" :y2="R" />
        <path class="shape" :d="path(panel.map)" />
        <circle class="marker" :cx="panel.map(marker)[0]" :cy="-panel.map(marker)[1]" r="0.12" />
      </svg>
      <figcaption v-html="panel.caption" />
    </figure>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { renderInline } from './quiz.js'

const props = defineProps({
  // W as rows: [[w11, w12], [w21, w22]]; a row vector x maps to x^T W
  matrix: { type: Array, required: true },
})

const R = 2.6
const ticks = [-2, -1, 1, 2]
// stem, top bar, middle bar
const strokes = [[[0, 0], [0, 2]], [[0, 2], [1.2, 2]], [[0, 1.1], [0.8, 1.1]]]
const marker = [1.2, 2]

const identity = (p) => p
const transform = (p) => {
  const [[a, b], [c, d]] = props.matrix
  return [p[0] * a + p[1] * c, p[0] * b + p[1] * d]
}

// SVG's y axis points down, so flip y when drawing.
function path(map) {
  return strokes.map((s) => s.map(map).map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${-y}`).join(' ')).join(' ')
}

const panels = computed(() => [
  { label: 'before', map: identity, caption: renderInline('rows $\\mathbf{x}_i^\\top$ of $\\mathbf{X}$') },
  { label: 'after', map: transform, caption: renderInline('rows $\\mathbf{x}_i^\\top\\mathbf{W}$ of $\\mathbf{X}\\mathbf{W}$') },
])
</script>

<style>
.linear-map-figure { display: flex; gap: 16px; justify-content: center; margin: 0 0 6px; }
.linear-map-figure figure { margin: 0; text-align: center; }
.linear-map-figure svg { width: 150px; height: 150px; background: #fff; border: 1px solid #c9d1d2; border-radius: 4px; }
.linear-map-figure figcaption { font-size: .72em; color: var(--nnds-ink-soft, #5f7073); }
.linear-map-figure .grid line { stroke: #e3e8e8; stroke-width: 0.02; }
.linear-map-figure .axis { stroke: #8a9799; stroke-width: 0.03; }
.linear-map-figure .shape { fill: none; stroke: var(--nnds-header, #23393c); stroke-width: 0.12; stroke-linecap: round; stroke-linejoin: round; }
.linear-map-figure .marker { fill: var(--nnds-accent, #8c0000); }
</style>
