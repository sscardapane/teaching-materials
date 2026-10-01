# Teaching materials

Read README.md before editing. This repository currently contains one pilot:
the linear-model lecture with NN and NNDS course variants.

- Edit shared lecture content once in `shared/slides/`; keep course notation,
  branding and layout choices in `courses/<course>/`.
- Preserve joint NN authorship and all source attribution.
- Build with `python3 scripts/build.py`. Validate structural changes with
  `python scripts/verify.py` in an environment with `requirements-verify.txt`.
- Inspect rendered PDFs when changing presentation content or layout.
- The baseline PDFs and provenance describe the October 1, 2026 experiment.
  Do not replace them merely to make a failing comparison pass. Intentional
  later content changes need a separately documented review and new baseline.
- Existing course repositories, public URLs and notebook sources remain in
  place during this pilot. Changes here do not authorize editing or deploying
  those repositories. Maintain one shared student/instructor notebook set.
- Keep generated intermediates in ignored `build/`. Stage explicit owned paths.
