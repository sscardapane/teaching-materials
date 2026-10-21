# NNDS normalization revision — 2026-10-21

Written by Codex (2026-10-21).

Simone requested implementing four recorded review points and sending screenshots:
show the BN smoothing claim, connect larger learning rates to the later argument,
clarify scale invariance and weight decay, and connect LN geometry to BN/LN
optimization. These are slide-specific edits to
`courses/nnds/Lecture_deeper_models.tex`; the existing visual system is retained.
The screenshots are for instructor review. No public PDF export is included.

## Upstream reconciliation

The first candidate pass used the local source at
`7323f4a01332bbea789368fd67da5a1036e72718` without checking the current remote.
Simone then pointed out his push. Read-only GitHub comparison showed the remote
was one commit ahead:
[`8c3ff6e098fa20ad7770c186bcd9702eeb9e02af`](https://github.com/sscardapane/teaching-materials/commit/8c3ff6e098fa20ad7770c186bcd9702eeb9e02af),
“refactor: reorganize slides on normalization scale invariance and weight decay mechanics”.

The candidates were reconciled onto that upstream file, rather than retaining
the stale order. Its source SHA256 is
`303881733e642a090eacd274d21fcb0807a35cbadc3d7a01b86cd5a22db63b03`.

- Kept “Why does BN help?” and its revised wording verbatim, including Simone's
  removal of the “full picture” sentence.
- Kept the upstream sequence: BN explanation; scale invariance; weight decay;
  learning-rate experiment; why normalize across features; “Understanding LN”.
  Added derivation slides within that sequence.
- Kept Simone's “Understanding LN” rewrite and geometry, with the epsilon
  approximation stated and paragraph spacing adjusted. Did not restore the
  numeric example he replaced.
- Expanded the scale and decay derivations without removing the gradient
  identities, squared-norm recurrence or any of the three consequences in
  Simone's version. The experiment's bridge now refers back to the derivation.
- Added the BN/LN comparison after “Understanding LN”.

The earlier screenshots are superseded by the reconciled set. No Git operation,
public PDF export or deployment was performed in this reconciliation.

## Dependencies and notation

| Explanation | Prerequisites already introduced | Notation and scope |
| --- | --- | --- |
| Smoothing evidence | Loss, gradients, gradient descent; preceding BN explanation | Full network parameters `theta`; `g = gradient L(theta)`; probes `theta - alpha g` |
| Scale cancellation | BN formula and its batch statistics | Positive scale `a`; training statistics; fixed gamma/beta; epsilon neglected |
| Gradient identities | `L(a w) = L(w)` | Data loss, chain rule in `a` and in `w`; pre-normalization weights used only through this path |
| Effective learning rate | Gradient orthogonality and inverse gradient scaling | `w = r u`, `r = norm(w)`, unit `u`; gradient descent without momentum |
| Weight decay | Effective learning rate and orthogonality | `lambda` is decay strength, `0 < eta lambda < 1`; L2 equivalence stated for gradient descent |
| BN/LN comparison | BN/LN axes, LN geometry, scale-invariant weight updates | BN per unit/filter; LN whole preceding affine map, bias included; CNN BN vector length `BHW` |

The LN optimization argument remains valid at the scale of a whole affine block.
It does not imply independent invariance of each incoming unit weight vector.
BN geometry concerns values across examples; LN geometry concerns a single
example's features. The smoothing evidence and the scale-invariance calculation
are complementary explanations, rather than a universal proof of learning-rate
stability.

## Coverage of the original source

The authoritative baseline for the reconciled pass is the upstream commit above:
21 explicit frames (24 PDF pages, including subsection dividers). The revised
deck has 25 explicit frames (28 PDF pages). Every upstream frame is retained in
its relative order; all upstream citations and figures remain.

The earlier local baseline SHA256 was
`e692545fcf903eca0557a01bb78fe2b5ac8bf0be6aa92b7a867d97df082d2ec5`.
The original preservation check against that file did not establish freshness.

| Upstream frame | Preservation status | Location or change |
| --- | --- | --- |
| Title page | leave | Unchanged |
| ImageNet and its challenge | leave | Unchanged |
| Going deeper on ImageNet | leave | Unchanged |
| From LeNet to AlexNet | leave | Unchanged |
| Can we just stack more layers? | leave | Unchanged |
| The rest of the lecture | leave | Unchanged |
| Normalizing the inputs | leave | Unchanged |
| Batch normalization | leave | Unchanged |
| Batch normalization in CNNs | leave | Unchanged |
| Batch normalization at inference | leave | Unchanged |
| How the running average works | leave | Unchanged |
| Fusing batch normalization at inference | leave | Unchanged |
| Layer normalization | leave | Unchanged |
| Layer normalization for images and for LLMs | leave | Unchanged |
| RMSNorm: dropping the mean | leave | Unchanged |
| Why does BN help? | leave | Upstream title and wording unchanged, with all three references |
| Normalized layers ignore the scale of their weights | substantial edit | Expanded cancellation and chain-rule derivations; original ray figure, identities and norm-growth result retained across the following slides |
| Weight decay as a learning rate | substantial edit | Explicit effective-rate and decay updates explain all three upstream consequences |
| Normalization allows larger learning rates | light edit | Curves, numbers and prose retained; bridge refers back to the effective-rate derivation |
| Why normalize across features? | leave | Upstream location and spacing retained |
| Understanding LN | light edit | Upstream rewrite and geometry retained; explicit epsilon approximation and paragraph spacing |

## Added frames

- **A more predictive gradient:** original Santurkar et al. Figure 4(a,b),
  with both measurement definitions and the interpretation of the shaded bands.
- **Scale invariance determines the gradient:** separate chain-rule calculations
  for gradient orthogonality and inverse scaling, plus the original ray figure.
- **The weight norm sets the effective learning rate:** exact normalized update
  and its small-step interpretation, deriving `eta_eff = eta / r^2`.
- **BN and LN: the same operation on different vectors:** shared shift/scale
  invariance; representation geometry versus per-unit or blockwise optimization.

## Figure provenance

`courses/nnds/assets/images/deeper/bn_smoothing.pdf` is a vector extraction of
Figure 4(a,b), PDF page 5, of Santurkar et al., *How Does Batch Normalization Help
Optimization?* (NeurIPS 2018): <https://arxiv.org/pdf/1805.11604v5>.
It is not a local reproduction or a redrawn curve. The crop retains both original
panels, including all axes, legends and panel labels. The adjacent panel (c) and
the paper's caption are outside the selected figure region.

`extract_bn_smoothing.py` records the source URL, crop coordinates and expected
source SHA256 `41d5195d959a8c37f3f23df34023032b2a0361887495cdb558a2402f60437d1e`.
Run it with the repository's Python environment containing `pypdf`.

The optimization derivation also uses van Laarhoven (2017),
<https://arxiv.org/abs/1706.05350>, with a direct chain-rule derivation through
the normalization statistics. The comparison uses Ba et al. (2016), Section 5.1
and Table 1, <https://arxiv.org/abs/1607.06450>.

## Verification

- Rebuilt the reconciled deck with the repository LuaLaTeX build script:
  28 PDF pages. The full NNDS build passed for all six registered decks.
- Command: `TEXMFCACHE=build/normalization-review-20261021/tex-cache TEXMFVAR=build/normalization-review-20261021/tex-cache python3 scripts/build.py nnds --outdir build/normalization-sync-render-20261021`,
  with both cache paths supplied as absolute paths in the actual invocation.
  The writable existing cache avoids LuaLaTeX's default cache restriction.
- Rendered and inspected all ten full slides in the reconciled explanation
  block (PDF pages 19-28) at 1600 pixels wide, including title, body, plots,
  equations, captions and footer. Eight replacement screenshots cover the
  revised or added slides, in narrative order.
- No overfull horizontal or vertical boxes on the revised slides. The two
  existing vertical warnings remain on the unchanged title page (29.87 pt)
  and Batch normalization (4.92 pt).
- Compared every upstream frame against the reconciled source: 17 are verbatim,
  four are edited as mapped above, all 21 retain their relative order, and every
  upstream citation remains. No conflict markers remain.
- Independently checked scale invariance, gradient orthogonality, inverse
  gradient scaling, the exact direction update and the norm update with decay
  using finite differences of a normalized linear model. Largest residual was
  below `4e-10`.
- Instructor acceptance of these candidates remains pending. Residual connections
  and the rest of the lecture are outside this revision.

## Final pass on “Why normalization works” — Claude, 2026-10-21

Written by Claude (2026-10-21). Polish on the reconciled candidate; no frame
added, removed or reordered (still 28 PDF pages); every citation kept, two added.

- **A more predictive gradient:** side labels now say what the bands show.
- **Normalized layers ignore the scale of their weights:** scaling stated in
  terms of the weights; the LN sentence now says invariance holds only for the
  whole layer (weights and bias) and points to the closing comparison.
- **Scale invariance determines the gradient:** figure label `-eta g` matches the text.
- **The weight norm sets the effective learning rate:** box reworded (same function,
  four times slower change of direction).
- **Weight decay as a learning rate:** Pythagoras step inlined to make room for
  the equilibrium result: balancing the two norm terms gives an angular update
  of about `sqrt(2 eta lambda)` per step, so eta and lambda act through their
  product. Derivation: `r^4 (2 eta lambda) ≈ eta^2 ||g(u)||^2`, so
  `eta_eff ||g(u)|| = sqrt(2 eta lambda)`. Added Li, Lyu and Arora (2020), NeurIPS.
- **Normalization allows larger learning rates:** “every rate from 0.005 to 1”
  corrected to every tested rate (0.002 also trains, `lr.csv`); the bridge now
  says that an oversized step grows the norm and lowers the effective rate
  (auto rate-tuning). Added Arora, Li and Lyu (2019), ICLR.
- **Understanding LN:** `z - mu 1`; the `r` in “radius r = sqrt(D)” dropped,
  since `r` means the weight norm on earlier slides; `sigma^2 = ||z_c||^2 / D`
  added to justify the radius; the invariance uses `a` and `\vc{}` like the rest
  of the section. Simone's wording otherwise kept.
- **BN and LN:** paragraph spacing inside the cards; the closing note now says
  concretely that LN has one effective rate per layer.

Build: same two pre-existing overflow warnings (title, BN at inference); all
revised slides inspected at full size.

## Smoothing evidence and section wrap-up — Claude, 2026-10-21

Written by Claude (2026-10-21). Simone was not convinced by the Santurkar figure
slide or the closing BN/LN slide, and chose our own measurement and a wrap-up.

- **A more predictive gradient:** the cropped Santurkar Fig. 4(a,b) is replaced
  by our measurement on the 20-layer plain CNN with and without BN (eta = 0.02,
  the depth-sweep optimizer and schedule). Every 100 steps: a probe step of
  0.05 along the gradient on 512 fixed training images. Scripts
  `smoothing_probe.py` and `smoothing_table.py`, data `smoothing.csv` (5-probe
  moving average). Raw runs are in the ignored `results/smoothing/`, including
  BN at eta = 0.1 (not plotted). After epoch 10 the gradient change ratio is
  7 to 20 times; the plain loss change turns positive from about epoch 6.
  Santurkar et al. stays as the source of the method. `bn_smoothing.pdf` and
  its extraction script stay in the repository but are no longer used.
- **Last slide:** “BN and LN: the same operation on different vectors” becomes
  “Why normalization works: BN and LN side by side”. It has a four-row table, a
  takeaway box and a bridge to residual connections (opener.csv: plain-bn 88.9%
  at 20 layers, 81.6% at 56). Dropped by Simone's choice: the description of
  BN's sphere as a batch and the remark that the geometry picture fits LN
  better. The invariance formula is on “Understanding LN”.

Still 28 PDF pages; same two pre-existing overflow warnings.

## Lighter text on the new slides — Claude, 2026-10-21

Simone found the new slides text-heavy, especially the experiment. The
experiment slide is now one setup line, full-width plots and a one-line
takeaway. Dropped from the slide, to be said aloud: that 0.05 is a rate the
plain network cannot train at; the measurement description; the detailed
box (loss rises without BN, gradient changes far less with BN). The wrap-up's
box and bridge are each one line, with the same content in shorter form.
- Same treatment, with Simone's OK, for “Weight decay as a learning rate” and
  “Normalization allows larger learning rates”. Dropped from the slides, to be
  said aloud: the Pythagoras formula, the hint ‖g(w)‖ = ‖g(u)‖/r, “other
  parameters can still be regularized”, the footnote (gradient descent,
  ηλ ≪ 1, equivalence with an L2 penalty), the setup and results text beside
  the learning-rate plot (failure above 0.02, every rate up to 1 trains with
  BN, best rate 5× larger), and “these are nominal rates”. The learning-rate
  plots are now full width.
