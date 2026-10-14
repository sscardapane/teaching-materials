# Derivatives and optimization: design pass

> [!note] Written by Claude (2026-10-14)

Simone approved the full proposal stored in the vault note *NNDS — Derivatives and
optimization design pass (proposal 2026-10-14)* ("change everything"). He asked
specifically for the momentum and Adam slides to be redone, so they no longer
keep their July layouts. Baseline: `5ae398f`, `courses/nnds/Lecture_2_preliminaries.tex`
(35 PDF pages). Result: 33 PDF pages. The PDF is not exported yet.

## Changes

- Course cards replace every plain gray box and the ad-hoc boxes on Descent
  direction, Adam and the convergence slide.
- Two sections (Derivatives · Optimization), four dividers: Derivatives and
  gradients · Gradient descent and convexity · Stochastic gradient descent ·
  Momentum and Adam (optional). Convexity now follows gradient descent.
- Notation: `∇f(x)` for the gradient, `∂f` for the scalar derivative and the
  Jacobian; function signatures `R → R`, `R^D → R`, `R^D → R^H` (chain rule
  `(O,D) = (O,H)(H,D)`); the angle in the descent argument is `α`, so it no
  longer clashes with the parameters `θ`; bold inputs in the ERM and SGD sums.
- `annotate-equations` is loaded again, restoring the labels on the first-order
  approximation and the iterative update. The first-order slide now gives the
  scalar form `f(x+h) ≈ f(x) + ∂f(x)h` before the vector form.
- Aligned layouts for Properties of derivatives and Examples of matrix
  derivatives; "A few additional definitions" became "Minima and stationary
  points" (`‖θ′ − θ‖ < ε`); "Finding stationary points" became "Iterative
  optimization"; "Properties of gradients" became "Chain rule for Jacobians";
  "Notable results" became "What gradient descent can guarantee" with three
  equal cards.
- Merged: the three SGD text slides into two; the two "Extracting batches"
  slides into one; the data-parallel caption moved into the body.
- New: Conditioning (elongated quadratic, `η < 2/κ`). Momentum and Adam are
  rewritten: momentum with the same quadratic showing GD against momentum and
  the unrolled-weights takeaway; Adam as three aligned updates with plain-language
  labels, a displayed bias correction and the per-coordinate takeaway. Muon removed.
- "Reading material" became "What comes next", pointing to 2c and to linear models.

Figures: `shared/assets/images/lecture_2/src/plot_conditioning.py` writes
`conditioning.pdf` and `momentum_path.pdf`. The derivative, Taylor, epoch and
data-parallel figures are unchanged, as decided in July.

## Verification

`python3 scripts/build.py --file courses/nnds/Lecture_2_preliminaries.tex` builds
with the pre-existing title-page warning only. Every slide was rendered and
checked at full size.
