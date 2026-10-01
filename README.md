# Teaching materials

Shared teaching sources for Simone Scardapane's courses. Course folders hold the
presentation templates and course-specific choices; shared material lives in one
place. The layout can accommodate Neural Networks, Neural Networks for Data
Science, and other machine learning courses.

The first experiment is the **linear-model lecture**, built for NN and NNDS.
All 23 content frames live in [one source](shared/slides/linear-models.tex).
Course settings preserve the existing notation and spacing differences. The
original title pages, templates, fonts, authorship and section dividers remain
part of each course wrapper.

## Pilot result

Both builds match fresh builds of the original repositories:

| Course | Original output filename | Pages | Different pixels at 200 dpi |
|---|---|---:|---:|
| NN | `NN2627_Linear_models.pdf` | 26 | 0 |
| NNDS | `Lecture_3_supervised_learning.pdf` | 29 | 0 |

Extracted text, page dimensions, link targets and positions, named destinations,
and title/author metadata also match. This verifies rendered equivalence in the
recorded environment; the PDF files themselves have different hashes because
build metadata and internal serialization can change.

See the [comparison report](verification/pilot-result.json) and
[baseline provenance](verification/provenance.json). The baseline PDFs were
compiled from committed source snapshots, including NN's pinned asset submodule.
The experiment does not fix layout issues already present in those sources.

## Build and compare

Use a full TeX installation with `latexmk`, pdfLaTeX and LuaLaTeX, including the
packages and fonts used by the two templates. The pilot used TeX Live 2024.
Poppler (`pdftoppm`, `pdftotext`) and Python 3.10+ are needed for verification.
Exact raster comparisons also depend on the same fonts and renderer version;
the report records the tool versions.

From the repository root:

```sh
python3 scripts/build.py
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-verify.txt
.venv/bin/python scripts/verify.py
```

Build one course with `python3 scripts/build.py nn` or
`python3 scripts/build.py nnds`. PDFs and build logs appear under `build/nn/`
and `build/nnds/`. The default verification report is `build/verification.json`;
a discrepancy returns a failing exit status. The committed pilot report is a
record of this experiment and is not overwritten by routine checks.

The build is self-contained: it does not read either original repository or
fetch a submodule. The committed figures are sufficient to compile; their
available drawing/plot sources are alongside them.

## Where to edit

- `shared/slides/linear-models.tex`: lecture content and sequence, shared by both courses.
- `shared/assets/`: figures and their available editable sources.
- `courses/nn/` and `courses/nnds/`: course wrappers, original templates, branding
  assets and `linear-models-settings.tex` for the few differing choices.
- `verification/reference/`: immutable baseline PDFs for this pilot.

Future courses can add their own folder under `courses/` and select the shared
material they need. New topics can have their own files under `shared/slides/`.

## Existing links and notebooks

This repository is a pilot. The existing course repositories and their published
URLs remain in place:

- [NN course repository](https://github.com/sscardapane/slides-nn-2026)
- [NNDS course repository](https://github.com/sscardapane/slides-nnds-2026)

PT01/PT02, including the instructor versions and student-generation scripts,
remain in the [shared notebook directory](https://github.com/sscardapane/slides-nnds-2026/tree/main/notebooks/drafts/2026-09-14-first-pass).
There is no second notebook copy here.

If the remaining material moves here, the old repositories can continue serving
the same filenames and URLs, receiving generated outputs from the shared source.
That publishing connection is a separate migration step; this pilot has no
deployment workflow.

## Attribution

The NN material preserves the joint authorship of Danilo Comminiello and Simone
Scardapane. The NNDS material preserves its original authorship and references.
The imported templates and assets retain their existing notices. Public
availability does not assign a new blanket license to third-party material.
