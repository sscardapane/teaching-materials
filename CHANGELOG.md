# Changelog

## 2026-10-13

> [!note] Written by Codex (2026-10-13)

- Fix NNDS builds failing with `fontspec` under pdfLaTeX by selecting the course's
  TeX engine in the VS Code / LaTeX Workshop recipe.
- Keep editor PDFs and auxiliary files in ignored `build/editor/courses/<course>/`
  and enable SyncTeX navigation. Command-line build and export defaults are preserved.
- Verified all eight command-line decks and both editor engines; the CNN and NN
  editor PDFs match their canonical build text and include SyncTeX data.
