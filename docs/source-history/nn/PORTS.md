# NNDS source tracking

## Selected source

- Repository: https://github.com/sscardapane/slides-nnds-2026
- Source and last reviewed commit: `44660a9ba39412ecd6f8a4daea1d812e659d97ad`.
- `upstream/nnds` is pinned to that commit. No working-tree changes were imported.
- The introduction and NN26 theme come from Simone's `sources.zip`; original
  checksums are in `template-source.json`. They remain unchanged.

## September 29, 2026 — Linear models and MLPs

Simone requested a port of both existing lectures before discussing extensions.
All 21 linear-model and 20 MLP content frames are retained in the original order.
NNDS title pages become joint-course NN title pages. NN26 generates section
outlines instead of NNDS's subsection dividers; the empty Introduction heading
in the linear-model source is omitted. No teaching content was removed or added.
NN lecture numbers and teaching dates are not inferred from NNDS numbering.

### Adaptations

- Both decks use the supplied `NN26` theme with joint authorship. `nn-content.tex`
  contains only supporting math/box helpers; it does not load NNDS's theme.
- Least squares uses ordinary matrix/component subscripts, and softmax uses
  ordinary output-component subscripts. No colored square-bracket indexing remains.
- The last MLP slide uses vectors x_t and y_t rather than matrix slices or the
  NNDS-only shape convention. Its diagram changes in the same way, using 1,2,3.
- The MLP reference to “Lecture 3” becomes “the linear-model lecture”.
- Existing advanced examples in the source MLP lecture are preserved. No new
  research extension from the earlier brainstorming has been added.

### Assets

Figures are loaded from `upstream/nnds/latex/images/lecture_3/` and `lecture_4/`.
The upstream native figure sources and Python generators remain there unchanged.
The build's `.fls` records identify exact input assets.
Unmodified TikZ diagrams: `residual.tex`, `swiglu.tex`, `untied.tex`, `shared.tex`,
`loop.tex` and `recurrence.tex` in upstream `latex/images/lecture_4/`.
The sole local adaptation is `latex/images/nn/positionwise.tex`, derived from
upstream `latex/images/lecture_4/positionwise.tex` at the selected commit.

### Prerequisites and labs

Linear models retains supervised-learning setup, data splits and loss functions;
MLPs follows linear models. Danilo's planned linear algebra/optimization coverage
is context, not confirmation of delivery. PT01 and PT02 were selected unchanged
on September 29. The October 1 decision supersedes this lab baseline: `LABS.md`
now links the shared notebooks on NNDS main, independently of the lecture-asset
pin. Item 25 adds optional PT01 tensor-semantics exercises at that shared source.

### Coverage ledger

Every entry below has preservation status **light edit**: the template changes,
with notation/cross-reference adaptations only where noted above.

#### `Lecture_3_supervised_learning.tex` → `latex/NN2627_Linear_models.tex`

1. Supervised learning
2. Training, validation, and test data
3. What is a good approximation?
4. Introducing loss functions
5. Expected risk and empirical risk
6. Overfitting
7. Losses for regression
8. A linear model
9. Least squares
10. Solving least squares
11. Regularizing least squares
12. Linear models for classification
13. From logits to probabilities
14. Softmax and temperature
15. Multiclass logistic regression
16. Binary classification
17. What makes the classifier linear?
18. Visualizing the sigmoid function
19. Cross-entropy from logits
20. Can we trust these probabilities?
21. Reading material

#### `Lecture_4_Fully_connected.tex` → `latex/NN2627_MLPs.tex`

1. Linear models are limited
2. The multilayer perceptron
3. Embedding the data
4. Adding hidden layers
5. Training the network
6. Universal approximation
7. Some terminology
8. Playing with a neural network
9. The rectified linear unit (ReLU)
10. ReLU as a gate
11. SiLU and GELU
12. Learning the negative slope
13. Learning the activation shape
14. Kolmogorov--Arnold networks
15. Combining branches
16. Input-dependent branch weights
17. Multiplicative blocks: GLU and SwiGLU
18. Sharing parameters and looping a block
19. Recurrence over a sequence
20. Shared MLPs on structured data

### Verification

Both decks compile with the documented pdfLaTeX/latexmk command (TeX Live 2024).
Full-slide renders were checked against both complete NNDS source decks and the
NN template. Outputs: 24 PDF pages for linear models, 22 for MLPs, including title
and section-outline pages. All content-frame titles and their order match the
source. Imported template checksums remain intact; the submodule is unmodified.
The supplied theme's title-page overflow/font-substitution, breakurl and unused
bibentry warnings remain baseline issues; rendered title content is visible.
No notebook execution is claimed: the labs are linked unchanged.

### Future updates

Review mapped upstream changes against NN adaptations before advancing the pin.
Record what was applied or intentionally deferred, including affected labs and
assets. A reviewed-but-deferred change is not an applied change.


## September 30, 2026 — Fixed nonlinear features in NN

User-requested NN revision, using the same pinned NNDS source revision as above.
The original September 29 coverage ledger records the initial port; the following
changes describe the current NN decks.

- Added two content frames immediately before “Reading material” in
  `latex/NN2627_Linear_models.tex`: “Linear in what?” combines XOR with the
  interaction feature x_1 x_2 and an explicit separating score; “Fixed nonlinear
  features” introduces a fixed feature map, polynomial feature growth, random
  Fourier features and kernel basis functions, then asks whether features can be
  learned. The Fourier-feature slide links Rahimi and Recht (2007).
- Reused and adapted the existing NN XOR TikZ diagram from the MLP opening,
  preserving its class colors and markers and adding coordinate labels.
- Updated the earlier “What makes the classifier linear?” transition to reflect
  that transformed inputs are now introduced in the current lecture.
- Removed “Linear models are limited” from `latex/NN2627_MLPs.tex` with the user's
  agreement. Its XOR example now appears in the linear-model conclusion. The
  MLP deck opens directly with “The multilayer perceptron”.
- The two closing slides and the removal of the MLP XOR opener were subsequently
  adopted in NNDS, preserving Simone’s revised wording (NNDS commit `a44f5cc`).
  The NNDS submodule pin and labs remain unchanged. Shared slide-source
  refactoring is deferred; course-specific wording and notation still matter.
- Both decks compile with pdfLaTeX/latexmk: 26 PDF pages for linear models (23
  content frames), 21 PDF pages for MLPs (19 content frames). Reviewed complete
  renders of both additions, the revised earlier transition and the MLP opening.
  No added overflow warnings remain. Existing title-page overflow and template
  font/breakurl/bibentry warnings are unchanged.


## September 30, 2026 — Generalization case study in NN

Added one instructor-reviewed content frame, “Training neural networks can be weird”, immediately
following “Playing with a neural network” in `latex/NN2627_MLPs.tex`, at the user's
request. The slide uses the paired loss/accuracy plots from Power et al. (2022),
Figures 4 and 1 (left), to show validation-loss recovery and delayed generalization.
It identifies division modulo 97 and a transformer; the original plots label the
50% training split. This is a published case study, not a local reproduction. The closing paragraph
names fitting random labels, model-size double descent and connected low-loss
solutions as topics outside this lecture, with links to primary sources.

Original figure images and a reproducible PDF extraction script live under
`latex/images/nn/grokking/`; its README records the source URL and hash. This is
also ported to NNDS after Simone’s final edits, including the explicit closing
line “Other surprising phenomena we have no time to cover”. Shared-source
refactoring remains deferred. Existing material and instructor edits are preserved.

Validation: pdfLaTeX/latexmk succeeds (22 PDF pages, 20 content frames). The full
new slide was rendered and checked for chart labels, citation visibility, margins
and fit. The final instructor-edited NN frame reports a small 2.18pt vertical
overflow; the rendered citation and all content remain visible. Existing
title-page/template warnings remain. A source comparison verifies that this edit only inserts the
new frame after the playground, without changing any pre-existing frame.


## September 30, 2026 — Session closed; selected additions in both courses

- Final NN outputs: 26 linear-model PDF pages / 23 content frames, and 22 MLP
  PDF pages / 20 content frames. Final NNDS outputs: 29 and 24 PDF pages, with
  the same respective content-frame counts.
- The NNDS grokking port preserves the instructor’s final prose and all four
  figure/provenance files. Only local font size, chart height, spacing and the
  displayed bibliography title were adapted to the NNDS theme. The new NNDS
  frame has no overflow warning; baseline warnings elsewhere remain.
- Every pre-existing NNDS MLP frame matches the pre-port source exactly. The
  grokking frame is inserted immediately after the playground in both courses.
- Keep the linear lecture extension brief. SVD/spectral-learning additions are
  deferred to next year; shared-source refactoring is deferred. No lab changes,
  model-size sweeps, random-label experiments or grokking reproductions were
  selected. Other menu proposals remain undecided.
- The submodule remains pinned to `44660a9`: these explicit reverse ports do
  not authorize importing unrelated upstream changes or changing lab versions.


## October 1, 2026 — ReLU geometry (NN only)

At Simone's request, added two content frames immediately after the ReLU
introduction in `latex/NN2627_MLPs.tex`:

- “One network, many affine maps”: a diagonal activation mask gives an exact
  affine map on each activation region. An original two-input, two-unit ReLU
  example shows the four regions, their gate patterns and output formulas.
  The general formula includes biases and the statement applies before any
  sigmoid or softmax at the output.
- “Depth through composition”: an explicit triangle made from three ReLUs,
  composed once, twice and three times, gives 2, 4 and 8 affine pieces on `[0,1]`.
  The comparison with the one-hidden-layer bound explains representation
  efficiency; a closing sentence recalls that purely affine layers collapse.
  Deep-linear training dynamics remain outside this addition.

This implements a compact selection from proposals 13 and 14, with only the
linear-composition contrast from 15. It is intentionally NN-only. NNDS and the
submodule pin are unchanged, as are all pre-existing NN frame contents and labs.

Original editable TikZ figures, provenance and an exact-arithmetic geometry
check live in `latex/images/nn/relu/`. Sources: Montufar et al. (2014) and the
triangle construction in Section 3.3 of Telgarsky (2016), linked on the slides.

Validation: `latexmk NN2627_MLPs.tex` succeeds, producing 24 PDF pages (22 content
frames). Full renders of the two additions and their transition into “ReLU as a
gate” were inspected at presentation scale. No new overflow warnings occur;
the baseline title-page and grokking-frame warnings remain. The figure check
passes for 6,561 rational inputs across the plotted regions, including shared
boundaries, and for triangle vertices, interior points and equivalent width-3
MLP realizations through eight compositions. Removing the insertion reproduces
the previous MLP source byte for byte. Existing untracked PDFs and `.vscode/`
files were left untouched.


### October 1 follow-up: smooth-activation transition

After the geometry addition, Simone noted that “ReLU as a gate” repeated the
binary-gate explanation. Renamed that NN frame “Smooth activation weights”,
removed the repeated hard-gate definition, and focused its prose on smooth
weights, the transition to SiLU/GELU, and the limit of the exact piecewise-affine
interpretation. The comparison plot and subsequent activation slides are unchanged.
Recompiled the 24-page deck and inspected the full revised frame; no new overflow
warnings. NNDS remains unchanged.
