# CNN opening review

> [!note] Written by Codex (2026-10-12)

## Scope and source

Simone approved the overall two-lecture split, then limited this session to the
first deck. Deep CNNs remain for later. Review proceeds section by section, using
the existing material as the baseline. This is a careful edit, not approval of a
complete rewrite.

The original source is
`NNDS_2024/6_Convolutional_Neural_Networks/Lecture_6_convolutions.tex`
in Dropbox. The editable course source is now
`courses/nnds/Lecture_CNNs.tex`. The title omits a lecture number because the
CNN/autodiff order is still open. The imported deck is registered in `decks.json`.

## Approved opening edits

| Original slide | Action |
| --- | --- |
| Definition of an image | Leave the definition and prose unchanged. |
| Vectorizing an image | Combine with the following three slides into one vectorize/unvectorize explanation. |
| Vectorization of an image | Replace the old figure with the new comparison figure. |
| Where has the image gone? | Replace the old figure with the new comparison figure. |
| Drawbacks | Fold reshaping, composition and parameter cost into Vectorizing an image. Remove the incorrect claim that flattening loses spatial information. |
| An example to keep in mind | Keep the prose and four-element matrix example. Adjust local spacing and matrix column separation so the complete matrix fits. |

Add **An MLP at every pixel** after the image definition, as a bridge from the MLP
unit. Add **Two ways to process an image** after the vectorize/unvectorize slide.
The comparison figure shows independent applications with shared parameters and
one global MLP between flattening and reshaping.

These are slide-specific edits to the opening. The resulting opening has five
content slides, on PDF pages 3 to 7. The complete source has 36 explicit frames
(37 in the original) and builds to 40 PDF pages, including dividers.

## Figures and preservation

The new figure's editable source is
`courses/nnds/assets/images/cnn/input-mixing.tex`; LaTeX embeds it directly in
**Two ways to process an image**. It uses the existing course font and colors.

`courses/nnds/assets/images/cnn/source-map.json` maps the ten retained figure
exports to their original paths and hashes. Available Draw.io/XML sources accompany
the exports. Some original exports have no corresponding native file in the source
folder; these remain unchanged. Header and university logo use existing course assets.

A source comparison verifies that everything from
`\section{Convolutional neural networks}` to the end remains identical after
normalizing figure paths. All ten imported figure exports match their original
SHA-256 hashes. No shared theme, other deck, notebook, HTML page or Deep CNN
material changed.

## Verification and review

`python3 scripts/build.py nnds` builds all NNDS decks successfully. The final CNN
source was rebuilt after the local spacing fixes. The opening and its original
source slides were rendered and inspected at full-slide size. The new and adjusted
content slides have no overfull box warnings. The title retains a template warning
with all content visible; later unreviewed source slides retain their existing
layout warnings.

The prose received the humanizer skill's final pass against the original opening
and the refreshed positionwise-MLP slide. The retained definition and example were
not rewritten.

Simone accepted the text slides and described the first comparison figure as
"honestly a bit weird". The replacement uses image grids: one selected pixel feeds
the positionwise MLP, while the complete image feeds the global MLP. The revised
figure builds without an overfull box warning and has been inspected as a full slide.
Simone asked for particular care with spacing and padding. The two row headings
now have equal clearance above their grids, arrow labels are centered, and the
MLP boxes have matching dimensions and padding. Simone accepted the opening
and asked to pause until October 13, 2026.

The accepted source and review record are saved as a Git checkpoint. No tracked
PDF was exported and no course-site deployment was triggered. Resume with
**Convolutional layers** in the first deck, continuing the careful section-by-section
review. Deep CNNs remain outside the current task.
