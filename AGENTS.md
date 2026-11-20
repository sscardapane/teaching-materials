# Teaching materials

Read README.md before editing. This is the canonical source for the migrated NN
and NNDS materials, including the shared PT01–PT05 lab notebooks.
The old course repositories are compatibility snapshots, not authoring targets.

- Edit shared linear-model and MLP content in `shared/slides/`; keep course
  notation, branding, optional content and layout choices in `courses/<course>/`.
- Colors, notation helpers and the card styles used by shared slides live once
  in `shared/style/content.tex`, loaded by both courses. Never redefine them in a
  course file; course themes own only header, footer, title page, fonts,
  `\myalert` and where `\SlideSource` text goes.
- After editing anything in `shared/`, run `python3 scripts/build.py all` and
  fix every entry under "Shared slides that overflow" so the slide fits in both
  courses.
- Preserve joint NN authorship, NN-only additions and source attribution.
- For NNDS narrative revisions, read `docs/NNDS_AUTHORING.md`. It describes
  instructor review, notation and visual checks; migration is not content approval.
- Edit notebooks in `notebooks/instructor/` only; the student and solutions
  notebooks are generated. Run `make_student_notebook.py` and `verify_notebooks.py`
  in `notebooks/`; use `--write-outputs` after intentional changes to refresh the
  instructor outputs and HTML previews. A solutions notebook is published only
  after its lab has taken place (`RELEASED` in `make_student_notebook.py`).
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
