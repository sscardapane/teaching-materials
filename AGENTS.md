# Teaching materials

Read README.md before editing. This is the canonical source for the migrated NN
and NNDS materials, including shared PT01/PT02 student and instructor notebooks.
The old course repositories are compatibility snapshots, not authoring targets.

- Edit shared linear-model and MLP content in `shared/slides/`; keep course
  notation, branding, optional content and layout choices in `courses/<course>/`.
- Preserve joint NN authorship, NN-only additions and source attribution.
- For NNDS narrative revisions, read `docs/NNDS_AUTHORING.md`. It describes
  instructor review, notation and visual checks; migration is not content approval.
- Edit notebooks in `notebooks/`, starting with `_solutions.ipynb`. Run
  `make_student_notebook.py --check` and `verify_notebooks.py` there; use
  `--write-outputs` after intentional changes to regenerate previews and sources.
- Build slides with `python3 scripts/build.py`; inspect changed full slides.
  `--export` updates tracked PDF downloads only after a reviewed revision.
- Run `scripts/verify.py` with `requirements-verify.txt` for source refactors.
  Reference PDFs describe the October 1, 2026 migration. Never replace them
  just to make a comparison pass. Intentional content changes need documented
  review and a separately identified new baseline.
- Build HTML with `npm ci` in `html/`, then `python3 scripts/build_site.py`.
  Check navigation, local assets and affected interactive controls.
- Historical documents under `docs/source-history/` preserve earlier decisions;
  paths, notebook locations and porting instructions there may be obsolete.
- Keep intermediates in ignored `build/` and `html/dist/`; stage explicit files.

Main-branch pushes affecting HTML, PDFs or notebooks deploy through the existing
Pages workflow. Publication is authorized for this migration; subsequent turns
follow their requested scope and the applicable standing approvals.
