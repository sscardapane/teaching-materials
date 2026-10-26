# Changelog

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
