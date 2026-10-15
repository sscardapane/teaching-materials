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

## Follow-up (same day)

Simone approved a second round: equation numbers removed throughout, "Optimizing
a function" removed (its max/min point is now one line on Empirical risk
minimization), the basis vector defined inline, and the momentum intro moved
above its columns. The 2c sub-deck's `pageOffset` is now `27`, so its pages
continue from 2b (last page 28); the old value `38` predated the July trims.

## Review fixes (2026-10-15)

Simone's review of the design pass:

- **Directional derivative:** the figure sat right under the definition box; it now
  has space above it. The gradient was already exactly normal to the tangent
  (equal axis aspect), but no drawn level curve passed through x, so the right
  angle was hard to see. The level set through x is now drawn darker, and the
  caption says "tangent to the level set".
- **Saddle points in more dimensions:** the reference was cramped against the body
  text; the figure is slightly smaller and there is space before the reference.
- **Descent direction:** columns are vertically centred, so the equation on the
  left no longer sits at the top with empty space below.
- **Conditioning, Momentum, Comparing the three methods:** the paths stopped
  after 15 steps, before the minimum. Each method now runs until it is within
  0.05 of the minimum: GD 46 steps, momentum 19, Adam 39 (step size 0.15, which
  removes the overshoot past the minimum). The comparison slide now says "running
  each method until it reaches the minimum".

Rebuilt both courses; only the pre-existing title-page warning remains on this deck.
Not exported.
