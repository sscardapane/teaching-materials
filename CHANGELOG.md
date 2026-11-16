# Changelog

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
