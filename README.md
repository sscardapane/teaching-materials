# Teaching materials

Shared sources for Neural Networks (NN), Neural Networks for Data Science
(NNDS), and future machine learning courses.

- [Shared PyTorch notebooks](notebooks/): PT01 and PT02, student and instructor versions.
- [NN PDFs](pdf/nn/) and [NNDS PDFs](pdf/nnds/).
- [Interactive slides and downloads](https://sscardapane.github.io/teaching-materials/),
  [slide sources](html/slides/) and [authoring guide](html/AUTHORING.md).

## One source, course-specific presentation

Linear models and MLPs use shared lecture sources in `shared/slides/`. Course
wrappers under `courses/nn/` and `courses/nnds/` preserve the original templates,
authorship, notation and spacing. The NN-only ReLU geometry material and the two
courses' distinct activation discussions remain explicit course choices. Course
introductions and NNDS preliminaries stay in their respective course folders.
Figures and their editable sources are in `shared/assets/` or the relevant
course's `assets/` directory.

PT01/PT02 have one shared home: **`notebooks/`**. Edit the `_solutions.ipynb`
instructor sources, then regenerate the student notebooks. The notebook README
covers dependencies, optional sections and verification. There are no dated draft
folders or separate notebook copies per course.

## Build slides

Use a full TeX installation with `latexmk`, pdfLaTeX, LuaLaTeX and the packages
and fonts required by the templates. The migration was checked with TeX Live 2024.

```sh
python3 scripts/build.py       # all seven LaTeX decks
python3 scripts/build.py nn    # NN only; use nnds for NNDS
```

Outputs and logs go to `build/<course>/`. After reviewing a deliberate slide
revision, add `--export` to copy the resulting PDFs into the tracked `pdf/`
download folders. The deck list is in `decks.json`.

To build the interactive site locally:

```sh
cd html
npm ci
cd ..
python3 scripts/build_site.py
```

The result is `html/dist/`, built for the `/teaching-materials/` URL prefix.
The script builds all four interactive decks and includes the PDF and notebook
downloads. Node 20+ is required. The tensor PDF is the preserved original HTML
export; its editable source is `html/slides/Lecture_2_tensors.md`.

## Publication

GitHub Pages builds the interactive decks and download index after main-branch
changes to `html/`, `pdf/`, `notebooks/` or the site builder/workflow. The workflow
is `.github/workflows/deploy-materials.yml`. It publishes to
https://sscardapane.github.io/teaching-materials/; NNDS interactive decks are under
`nnds/<deck-name>/`. PDF downloads also remain available directly from `pdf/` in
this repository. A source-only TeX edit must be reviewed and exported before its
public PDF changes. Verify the Actions deployment and affected live URLs.

## Migration checks

All seven LaTeX decks were rebuilt from independent snapshots of the original
repositories and compared with the migrated sources. The
[comparison report](verification/migration-result.json) records every page at
200 dpi, extracted text, page dimensions, PDF links and metadata. The original
[linear-model pilot](verification/pilot-result.json) remains as a historical check.

The subsequent [NN box-spacing review](verification/nn-box-layout-review.json)
records an intentional layout change on nine NN slides: boxes fit their contents
while adjacent boxes keep equal heights. The two PDFs in `pdf/nn/` and their
recorded hashes identify that reviewed layout baseline. The migration comparison
above still targets the untouched original PDFs, so it now reports those nine
NN pages as expected differences; the NNDS decks remain identical.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-verify.txt
.venv/bin/python scripts/verify.py
```

Exact pixel equality requires the same TeX/font/Poppler environment; tool
versions are recorded in the reports. PDF bytes may differ because of build
metadata. Existing layout limitations are preserved by the migration.

The [inventory](verification/migration-inventory.json) accounts for every tracked
file in both original repositories. Generated TeX intermediates and editor/Git
configuration were replaced by the new build setup. Original course decisions,
port history and provenance are retained in `docs/source-history/` as historical
records. They are not current operating instructions.

All five previously published NNDS PDFs and all eight notebook/HTML files were
copied byte for byte; see [download checks](verification/preserved-downloads.json).
The public PDFs are publication snapshots. In particular, the old preliminaries
PDF has a small equation-spacing difference from a fresh source build. This
migration preserves the existing download; a future reviewed export can update it.
The three NN downloads come from the verified migrated builds.

## Existing URLs

`teaching-materials` is now the source to edit. The previous repositories retain
their material for history and compatibility, so existing raw PDF, notebook and
interactive-slide URLs continue to resolve. They are frozen snapshots and are
not independently maintained teaching sources. New links should use this
repository's paths. Original Google Slides and Colab documents remain external
sources; the migration does not copy or replace them.

## Attribution

NN preserves the joint authorship of Danilo Comminiello and Simone Scardapane.
NNDS preserves its original authorship and references. Imported assets and
source notices remain intact. Public availability does not assign a new blanket
license to third-party material.
