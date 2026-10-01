# Neural Networks material guidance

Read [NN_MATERIALS.md](NN_MATERIALS.md) before porting, editing or updating material.
This is the private NN 2026–2027 course repository, co-taught by Danilo Comminiello
and Simone Scardapane. Preserve the supplied NN26 theme and both instructors’ work.

NNDS is an upstream content source, not this course’s schedule or teaching policy.
Use [PORTS.md](PORTS.md) to identify imported source versions and intentional NN
changes. Never bulk-overwrite NN from NNDS or change NNDS as a side effect.

Run `latexmk <deck.tex>` from `latex/` (pdfLaTeX); output belongs in `build/`.
Validate changed notebooks by executing them in a clean kernel where feasible.
Review complete rendered slides, not compilation alone. No deployment is configured.

Unchanged lecture assets are referenced through the pinned `upstream/nnds`
submodule; initialize it with `git submodule update --init upstream/nnds`.
NN-specific asset adaptations live in `latex/images/nn/` with provenance.
Use ordinary subscripts/named vectors instead of NNDS's colored bracket indexing.
PT01 and PT02 have one shared source for all courses in NNDS; see `LABS.md`.
Edit that source for approved lab additions, without making course-specific copies.
Lab links follow NNDS main independently of the pinned lecture assets.
