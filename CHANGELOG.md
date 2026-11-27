# Changelog

## 2026-11-27

> [!note] Written by Claude (2026-11-27)

- Sinusoidal positional embeddings in `shared/slides/transformers.tex` (NN and
  NNDS) rewritten around the clock analogy, after Simone's lectures of
  November 27: *A clock with one hand* (a TikZ clock read as [sin ωt, cos ωt]),
  *A clock with many hands* (geometric ω_j, base b chosen from the longest
  sequence; the book plot is kept), *The sinusoidal embedding matrix* (new
  figure from `sinusoidal_embeddings.py`: ω_j in log scale and E for D=64),
  *Shifting the position rotates the hands* (two clocks, each hand turned by
  ω_jp) and *Shifts as a linear map* (E_{t+p}=E_t T(p)). They replace
  *Sinusoidal embeddings* and *Sines and cosines*.
- RoPE (four slides) and *Interleaving RoPE and NoPE* moved to the new
  `shared/slides/rope.tex`. NNDS teaches them in `Lecture_modern_transformers`
  (new subsection after SwiGLU); NN keeps them in its transformer deck, after
  NoPE. Not exported.
- Same day, at Simone's request: the whole relative part moved, not only RoPE.
  `rope.tex` became `shared/slides/relative-positions.tex` (relative
  embeddings, RoPE, ALiBi, NoPE, RoPE/NoPE interleaving, summary table). The
  first NNDS deck now ends its positional section with the sinusoidal shift
  T(p); NN keeps the whole block right after it.

## 2026-11-25

> [!note] Written by Claude (2026-11-25)

- New NNDS deck `Lecture_recurrent_models` (draft, not exported), refreshed
  from the 2024 Lecture 10 in the same order. New opening on the KV cache and
  sliding windows; *Linearized attention* and *Kernel functions* kept; two new
  code slides (causal linear attention in parallel and recurrent form, a
  for-loop RNN). The SSM part is shorter: one convolutional-view slide, the
  scan example on one slide, two diagonal-SSM slides. Corrected: linear
  attention costs O(TED_v) (not O(n(e²+ev))), the scan takes O(log n) steps and
  O(n) work, each scan product costs O(E³), B̄ = P⁻¹B, Mamba keeps the scan but
  loses the convolutional view. Book figures (recurrence, BPTT, associative
  scan, Mamba) copied with their draw.io sources to
  `courses/nnds/assets/images/recurrent/`. Sections on modern linear
  recurrences and hybrid models follow.

## 2026-11-20

> [!note] Written by Claude (2026-11-20)

- Lab notebooks redesigned for 2–3 hour sessions (Simone's plan of November 20):
  a core of a few activities with one-line checks (`check_conv(MyConv2d)`), the
  rest moved to optional sections, and a solutions-only notebook per lab
  (heading, short explanation and code for each exercise).
- The old PT04 Part 2 is now `PT05_Training_deeper_networks`. PT04 keeps the
  creation of the convolution weights (stride 1, padding as a parameter; stride is
  Optional A). PT05 keeps the full batch-normalization layer with running
  statistics; its check accepts the biased or the unbiased running variance.
- PT02: Activities 4–6 become Optional A–C, Section 9 becomes Optional D; the
  fallback wrappers (`run_step`, `forward_logits`) are gone. PT03: `matmul_vjp`
  handles two matrices; the `@` operation turns vectors into matrices.
- Instructor sources moved to `notebooks/instructor/` (not copied to the Pages
  site); `PTxx_*_solutions.ipynb` is now the generated solutions notebook.
  `make_student_notebook.py` builds both notebooks for every lab from cell tags
  (`reference`, `solution`, `exercise`); `verify_notebooks.py` and
  `verify_regressions.py` replace `verify_autodiff.py` and `verify_cnn.py`.

## 2026-11-19

> [!note] Written by Claude (2026-11-19)

- Transformer decks (NN and NNDS, `shared/slides/transformers.tex`): new opening
  subsection *Sequence-to-sequence models*, from Simone's blackboard
  introduction. *Sequence to sequence* (machine translation with an encoder,
  pooling and a causal decoder, with a new TikZ figure), *From pooling to
  attention* (Bahdanau et al.), *Transformers* reworded to explain the title
  and the encoder/decoder names, and *Sequence to sequence today* (GPT-3
  prompting, prefill as what remains of the encoder). *Encoder-decoder models*
  now points back to the opening. Not exported.

## 2026-11-16

> [!note] Written by Claude (2026-11-16)

- Add the NN deck `NN2627_Transformers` (tokens, embeddings and transformers),
  not yet exported. Its slides are shared with NNDS: `shared/slides/text-embeddings.tex`
  (the text pipeline frames of the CNN deck) and `shared/slides/transformers.tex`
  (the body of `Lecture_transformers`). `\NNDSorNN{NNDS text}{NN text}`, defined
  by each course, holds the few sentences that differ. NN uses conventional
  subscripts for `\idx` and the real equation annotations.
- Move the transformer colors and diagram styles and `\shape` into
  `shared/style/content.tex`, and the transformer figures and the four text
  figures into `shared/assets/images/`. All NNDS and existing NN decks render
  pixel-identical to their previous builds.
- NN26 theme: skip the section-page progress bar on the first pass, when the
  total frame count is still unknown; decks longer than about 46 frames
  overflowed the length and failed to build.
- `scripts/verify.py` skips decks added after the migration (no reference PDF)
  instead of stopping at the first one.
- Add a *Language models* section to `NN2627_Transformers`, after the text
  pipeline: the CNN deck's next-token prediction, training, probabilistic view,
  autoregressive generation and decoding frames, now shared as
  `shared/slides/next-token-prediction.tex` and `shared/slides/language-models.tex`
  (their three figures move to `shared/assets/images/text/`). The NN wording
  says explicitly that this repeats the time-series autoregression of the
  recurrent lectures for text. The NNDS CNN deck renders pixel-identical.
- NN transformer flow: NN students arrive from recurrent models, not from the
  NNDS sequence-convolution slides. A new `\ifNNcourse` switch (both courses)
  selects whole frames. In NN, *Beyond recurrence* (long paths through the
  state, sequential computation) replaces *Beyond locality*; the 1D convolution
  is introduced from the image convolutions and defines locality; the block
  design no longer assumes 1×1 or depthwise-separable convolutions; the
  computational comparison adds a recurrent layer. NNDS decks render
  pixel-identical.

## 2026-11-06

> [!note] Written by Claude (2026-11-06)

- Simplify PT03 at Simone's request. The `Tensor` now follows PyTorch
  (`data`, `requires_grad`, `.grad`, and a `backward()` method that accumulates
  into the `.grad` of leaves) and gains its operations section by section.
  `sum_to_shape` and `topological_order` are given. The verifier's rejection
  checks follow the new API.

## 2026-10-26

> [!note] Written by Claude (2026-10-26)

- Add PT04, convolutional neural networks from scratch on KMNIST, adapted from
  Simone's CNN Colab lab: student and instructor notebooks, HTML previews and
  `verify_cnn.py` (run alone with `verify_notebooks.py --only-cnn`). KMNIST is
  downloaded to the untracked `notebooks/kmnist/`, outside the published data folder.
- Rate the exercises of PT01, PT02 and PT03 with the quiz's 1 to 3 stars.

## 2026-10-13

> [!note] Written by Codex (2026-10-13)

- Fix NNDS builds failing with `fontspec` under pdfLaTeX by selecting the course's
  TeX engine in the VS Code / LaTeX Workshop recipe.
- Keep editor PDFs and auxiliary files in ignored `build/editor/courses/<course>/`
  and enable SyncTeX navigation. Command-line build and export defaults are preserved.
- Verified all eight command-line decks and both editor engines; the CNN and NN
  editor PDFs match their canonical build text and include SyncTeX data.
