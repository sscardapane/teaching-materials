# CNN lecture: final design refresh

> [!note] Written by Claude (2026-10-14)

## Agreed scope

Simone's design brief of 2026-10-14 (stored in the vault note *NN and NNDS — MLP
design pass and CNN review*) is the approved ledger. On the same day he asked for
the final refresh to implement all of it, including the four possible additions.
The baseline is `196b261`, `courses/nnds/Lecture_CNNs.tex` (39 frames, 46 PDF
pages). The result has 40 frames and 45 PDF pages, with four subsection dividers
instead of six. No shared styles were edited, and the PDF is not exported yet.

Dilated convolutions return as a single slide. An earlier pass had removed dilation
to streamline the lecture; the brief brings it back as an addition.

## System-level changes

- **Cards.** Plain gray boxes became course cards: blue for local layer,
  translation equivariance, receptive field, causal model, CNN parts and landmark
  architectures; green for the padding rule, the downsampling takeaway, the
  invariance note, data augmentation and exponential receptive-field growth;
  orange for the label-preservation warning.
- **Notation.** Images are `X ~ (H, W, C)` with `C'` output channels; layer
  outputs are `Y` (matching the MLP slide's `Y[t,:]`). Sequences are `(T, C)`,
  embedded text `(T, D)`, positions `t`. The implementation slide states that
  PyTorch is channels-first `(B, C, H, W)` while the math is channels-last.
- **Indexing.** Zero-based throughout: running example `x_0..x_3`, weights
  `W_00..W_33` and `W_0..W_2`, token IDs in `{0, ..., V-1}`.
- **Dividers.** Images and MLPs · Convolutional layers · Building a CNN ·
  Sequences, text and generation.
- **Title.** "Convolutional networks: from images to text generation".

## Coverage of the original frames

| Original frame | Status | Note |
| --- | --- | --- |
| Definition of an image | light edit | No `\stackunder`; `(H,W,C)` |
| An MLP at every pixel | light edit | Notation |
| Vectorizing an image | light edit | Notation |
| Two ways to process an image | light edit | Figure label `Y[i,j,:]` |
| An example to keep in mind | light edit | Zero-based indices |
| Local neighborhoods | light edit | Blue card |
| Convolutional layer | light edit | Notation |
| Filters and output channels | light edit | Notation |
| Zero-padding | light edit | Green card, zero-based |
| Running example | light edit | Zero-based |
| Translation equivariance | light edit | Blue card |
| Receptive field | light edit | Blue card |
| Implementation | light edit | Channels-first line |
| Reducing dimensionality | substantial edit | Receptive-field motivation; "multi-resolution filter banks" dropped |
| Stride | light edit | "zero-based" remark dropped |
| Convolution with a larger stride | substantial edit | cs231n screenshot replaced by `assets/images/cnn/stride.tex` |
| Max-pooling + Visualization of max-pooling | merged | One slide with the 2×2 formula and the existing figure |
| Complete architecture (1) + (2) | merged | Backbone / head / classifier cards; invariance in a green card; raw `\color{blue}` removed |
| Example of a simple CNN specification | light edit | Draw.io source redrawn in two rows (backbone, then head and classifier) with horizontal, larger labels |
| Designing in blocks | light edit | Draw.io source redrawn; caption grammar fixed |
| Spatial mixing and channel mixing | light edit | "Each layer of a positionwise MLP is a 1×1 convolution"; depthwise output named `Y` |
| Temporal sequences | light edit | `(T, C)` |
| 1D convolutional layers | light edit | `(T, C)`, position `t` |
| Handling text data | light edit | Step 3 shortened to a forward pointer |
| Text tokenization | leave | |
| Subword embeddings | light edit | `E ~ (V, D)` |
| Byte pair encoding | leave | |
| Dense embeddings | light edit | Bias sentence added for the Bolukbasi citation |
| Visualizing the embedding procedure | leave | |
| Sequence padding | light edit | `(T, D)` |
| Two tasks on a sequence | light edit | Position `t` |
| Masked convolutions + Causal convolutions | merged | Causal card; left-only padding with `s-1` zeros; mask kept as one line; figure redrawn with labels and the padding zero, placed beside the layer equation |
| Training a next-token predictor | light edit | Position `t` |
| Autoregressive models | light edit | Temperature sentence closing the linear lecture's promise |
| Autoregressive generation | leave | |
| Further readings | substantial edit | Became "What comes next": Chapter 7, normalization and residuals next, then attention |

## Additions

| New frame | Position | Source |
| --- | --- | --- |
| What filters learn | after Filters and output channels | `assets/images/cnn/filters/make_figures.py` (Sobel filter on the public-domain scikit-image astronaut) |
| Data augmentation | after Complete architecture | same script |
| A few landmark CNNs | after Spatial and channel mixing | LeNet-5, VGG, ConvNeXt; ResNet as a forward pointer |
| Dilated convolutions | after Causal convolutions | `assets/images/cnn/dilated.tex` |

## Figures

The three draw.io sources (`book/CNN_architecture.xml`, `book/CNN_blocks.xml`,
`sequences/Causal_convolutions.xml`) were redrawn as uncompressed draw.io files
and exported with draw.io's own export page (the desktop CLI does not start
headless on the Mac Studio). `source-map.json` still records the migrated
originals and was not updated. `source/stride.jpg` is no longer used.

## Verification

`python3 scripts/build.py --file courses/nnds/Lecture_CNNs.tex` builds with one
overfull warning, on the title page, as before. Every new, merged or redrawn
slide was rendered and checked at full size for overflow, overlap and footer
collisions.

## Review fixes and export (2026-10-15)

After Simone's own edits (`2129f13`), his review asked for five changes:

- **Two ways to process an image:** the diagram is redrawn around what one output
  pixel depends on. In the positionwise MLP it sees one input pixel; in the
  vectorized layer it sees all of them (one wire per pixel). Each row states the
  weight count, and a closing line sets up convolutions as the middle ground.
- **Byte pair encoding:** step 2 now says each merge is *added to the vocabulary*,
  and a line shows the toy vocabulary growing from the characters to `ug` and `hug`.
- **Dilated convolutions:** the figure is redrawn with edges under the units, one
  label per layer, the output marked, and a brace for the 8-position receptive
  field. The box text is shorter.
- **Training a next-token predictor:** the IDs (2, 16, 3) were never defined.
  The example is now a table by position: input `z_t` and target `z_{t+1}`, with the
  tokens next to their IDs.
- **Autoregressive generation:** the caption defines the hat: `x̂_{t+1}` is the model's
  prediction, fed back as the next input.

The deck loads `decorations.pathreplacing` for the brace. Exported with 2b; baseline
`verification/export-2b-cnn-2026-10-15.json`.

## Post-lecture fixes (2026-10-30)

> [!note] Written by Claude (2026-10-30)

After the lecture, Simone asked for six changes, all before "Sequence padding"
(that slide and everything after it are untouched):

- **Title:** "Lecture 5: Convolutional models".
- **Example of a simple CNN specification:** the return arrow ran along the
  Backbone border; it now turns outside the box (draw.io source and PDF).
- **Convolution with explicit indices (new):** the 2D layer with
  `K ~ (s,s,C,C')` and the 1D layer with `K ~ (s,C,C')`, with the spatial sums in
  blue and the channel sum in green. It restores the 2024 "Equivalent
  formulation" slide in zero-based form.
- **Spatial mixing and channel mixing:** the pointwise layer is now written with
  indices too; each equation keeps only its colored sum. The separable and group
  convolution sentences moved to a new **Depthwise-separable convolution** figure
  slide, which adds the weight count `s^2C + CC'` against `s^2CC'`.
- **Notation:** matrices are bold uppercase (`X ~ (T,C)`, the 1D window and
  output, `U(X)`, `Z`, the edge filter, the embedded rows).
- **A 1D convolutional layer (new):** a figure of the window over a
  `(T,C)` matrix and the output row in `Y ~ (T,C')`.
- **Text:** "Handling text data" is now a three-step pipeline figure on
  `Check this out` (IDs 2, 16, 3, as in the next-token slides). New slides
  "Step 1: choosing the tokens" (real GPT-4 tokenizer output for
  `Classify this text!`), "Step 1: building the vocabulary" (three cards) and
  "Step 2: from tokens to IDs" (vocabulary lookup with an unknown token). The
  subword and BPE slides are unchanged. "Dense embeddings" became "Step 3: dense
  embeddings". The old tokenization and embedding figures were replaced by
  editable TikZ figures; their PDFs stay in `assets/` for the old URLs.

Not exported yet: the tracked PDF waits for Simone's review of the new slides.

Simone then asked to extend the bold-matrix fix to the rest of the deck: the
embedded matrices in "Sequence padding", `X` and `Y` in "Two tasks on a
sequence" and "Causal convolutions", and the causal window `P` (also used for
the dilated window). The padded batch `X ~ (2,T_1,D)` stays a non-bold tensor.
