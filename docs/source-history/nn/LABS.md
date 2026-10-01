# PyTorch labs

PT01 and PT02 are maintained as one shared set for all courses, including NN
and NNDS (Simone's decision, October 1, 2026). Lectures select the sections they
need from these notebooks. Do not create NN-specific copies.

- [PT01: Introduction to PyTorch](https://github.com/sscardapane/slides-nnds-2026/blob/main/notebooks/drafts/2026-09-14-first-pass/PT01_Introduction_to_PyTorch.ipynb)
- [PT01: Instructor solutions](https://github.com/sscardapane/slides-nnds-2026/blob/main/notebooks/drafts/2026-09-14-first-pass/PT01_Introduction_to_PyTorch_solutions.ipynb)
- [PT02: Logistic regression](https://github.com/sscardapane/slides-nnds-2026/blob/main/notebooks/drafts/2026-09-14-first-pass/PT02_Logistic_regression.ipynb)
- [PT02: Instructor solutions](https://github.com/sscardapane/slides-nnds-2026/blob/main/notebooks/drafts/2026-09-14-first-pass/PT02_Logistic_regression_solutions.ipynb)
- [Dependencies, data and verification](https://github.com/sscardapane/slides-nnds-2026/blob/main/notebooks/drafts/2026-09-14-first-pass/README.md)

These links follow the shared notebooks on NNDS `main`. The directory retains
its draft name for stable links. Edit that source for future lab additions.
Both notebooks have student and instructor variants, shared across courses. Edit
the instructor sources and rebuild the student notebooks with the upstream
`make_student_notebook.py`; exercise answers and executable reference solutions
stay in the instructor variants. PT02 includes an optional per-example-gradient
demonstration and a coordinate-to-RGB reader exercise for after the MLP lecture.

The `upstream/nnds` submodule remains pinned for lecture assets. Its historical
notebook files are not the current lab source; lab revisions do not advance the
slide-asset pin. For local execution, use the shared NNDS checkout and its
notebook dependencies and bundled data.
