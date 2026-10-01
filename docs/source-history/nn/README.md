# Neural Networks 2026–2027

Private teaching material for the course co-taught by **Danilo Comminiello** and
**Simone Scardapane**, Sapienza University of Rome.

The instructors alternate topics. The course stays linear and self-contained,
with PyTorch as its main practical framework. NNDS provides reusable material;
NN has its own template, pacing and advanced additions.

## Files

- `latex/NN2627_01_Intro.tex`: supplied introduction, unchanged.
- `latex/beamerthemeNN26.sty` and `latex/images/`: supplied theme and assets, unchanged.
- `template-source.json`: archive provenance and checksums of the imported files.
- `NN_MATERIALS.md`: agent workflow for ports, maintenance and course discussions.
- `latex/NN2627_Linear_models.tex` and `latex/NN2627_MLPs.tex`: NNDS lectures
  ported to the joint NN template, with conventional indexing.
- `upstream/nnds/`: pinned NNDS submodule for unchanged assets and Python sources.
- `latex/images/nn/positionwise.tex`: the one diagram adapted for NN notation.
- `LABS.md`: links to the shared PT01 and PT02 notebooks.
- `PORTS.md`: source versions, adaptations, coverage and verification.
- `build/`: ignored local PDFs and build intermediates.

## Build

With TeX Live and latexmk installed:

```sh
git submodule update --init upstream/nnds
cd latex
latexmk -interaction=nonstopmode -halt-on-error NN2627_01_Intro.tex
latexmk -interaction=nonstopmode -halt-on-error NN2627_Linear_models.tex
latexmk -interaction=nonstopmode -halt-on-error NN2627_MLPs.tex
```

The local configuration uses pdfLaTeX. It disables the introduction’s unnecessary
BibTeX run; for a future deck with bibliography data, use `latexmk -bibtex <deck.tex>`.
The supplied deck compiles to 32 pages with TeX Live 2024. Existing font-substitution,
bibentry and overfull-box warnings are baseline issues, not a visual-quality approval.
Preserve the source until a content or layout revision is requested.

The two lecture PDFs are produced in `build/`. No Pages deployment or public
course publishing is configured. When cloning, `git clone --recurse-submodules`
initializes the asset dependency in one step. GitHub's source ZIP does not include
submodule contents; use Git or obtain the exact NNDS revision listed in `PORTS.md`.

Unchanged assets stay in NNDS, including their editable originals and generating
scripts. The submodule stores a commit reference in this repository; a local clone
still downloads its contents. Pinning makes builds reproducible while allowing
reviewed upstream updates later. Only modified NN assets get local copies.
