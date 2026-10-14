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
