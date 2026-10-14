# CNN defining-the-network review

> [!note] Written by Codex (2026-10-14)

## Scope

Simone resumed the CNN refresh and said the next section could remain more or
less similar. This candidate keeps the nine slides of **Defining the network**
in their original order, with all four original figures. The authoritative
source is `courses/nnds/Lecture_CNNs.tex`. It follows the opening and
convolutional-layer checkpoints recorded in this directory.

The edits are local to this subsection. Its slides remain on PDF pages 18 to 26
(printed numbers 15 to 23). The complete deck still has 26 explicit frames and
30 PDF pages, including dividers. The accepted opening, the preceding
convolutional-layer candidate and everything from **Other notable types of
convolutions** onward are unchanged.

## Original-slide coverage

| Original slide | Status | Result |
| --- | --- | --- |
| Reducing dimensionality | light edit | Retain the block composition and multiresolution connection. Qualify spatial-size preservation by the previously introduced padding and evaluation at every position. |
| Stride | light edit | Use r for stride, retaining s for kernel width. Match P_k(i,j), W^T and bias from the previous section; use zero-based indices. Give the output shape with rounding and qualify exact halving by even input dimensions. Replace TensorFlow with nn.Conv2d. |
| Convolution with a larger stride | light edit | Retain the original figure and scale. Shorten its caption and preserve its CS231n attribution as a link. |
| Max-pooling | light edit | Keep maximum/average aggregation, independent channels and the 2x2/4x4 examples. State the window and stride needed to halve even spatial dimensions. |
| Visualization of max-pooling | light edit | Keep the original figure and scale. Clarify that the illustrated window and stride are both 2; retain the alternative-aggregation remark. |
| Complete architecture of a CNN (1), previously titled only (1) | light edit | Restore the title, allow either pooling or strided convolution for downsampling, and qualify translation invariance. Retain the classifier and model-selection discussion. |
| Complete architecture of a CNN (2) | light edit | Keep the four-row mathematical architecture, shapes, classifier example and meaningful equation colors. Explicitly identify flattening and global average pooling as alternatives. Complete the channel slice and adjust local array spacing to remove overflow. |
| Example of a simple CNN specification | light edit | Keep the original figure. Correct the claim that repeated downsampling is required before classification: global pooling gives one value per channel. Slightly reduce the displayed width and remove surplus top spacing so the caption clears the footer. |
| Designing in blocks | light edit | Keep the original figure and scale. Correct its caption and reduce local top spacing. |

No slides, definitions or figures are removed, added or moved. All four figures
retain their source files and provenance. No shared style, notebook, other deck,
HTML page or published PDF changes form part of this candidate.

## Dependencies and notation

| Slide | Prerequisites |
| --- | --- |
| Reducing dimensionality | Convolution, activation, composition and padding from preceding sections. Avoid using stride before its definition on the next slide. |
| Stride | Local tensor P_k(i,j), kernel width s=2k+1, shared W^T and bias. Introduce r before using it. |
| Max-pooling and its figure | Image grid and stride. |
| Architecture overview | Convolution, downsampling, pooling and earlier translation equivariance. |
| Architecture equation and example | Flattening and global pooling from the preceding slide; softmax and the classifier from linear models. |
| Designing in blocks | Composition and the complete example. |

For contiguous odd kernels with k=(s-1)/2 padding on each side, a stride-r
convolution has spatial shape (ceil(h/r),ceil(w/r)). Its neighborhoods are centered
at (ri,rj), using zero-based indices. The stride symbol does not change the
previous meaning of s. Mathematical tensors remain channel-last; the preceding
implementation slide introduces PyTorch's channel-first convention.

Global average pooling preserves a channel's average when a shift merely
rearranges the spatial entries. Downsampling and image boundaries can prevent
exact translation invariance for the whole CNN. This qualifies the earlier
approximate claim without adding a separate anti-aliasing discussion.

## Verification and review

`python3 scripts/build.py nnds` builds all five NNDS decks. The final nine-slide
section has been rendered and inspected in full at 1600-pixel width against the
original slides. It has no overfull box warnings. The deck title retains its
pre-existing warning; the later unreviewed slides are unchanged.

Source checks confirm nine frames and the same four figure references in the
same order. They also confirm that all source outside this subsection is
identical to the starting checkpoint. All ten imported CNN figure exports match
their recorded SHA-256 hashes. The diagrams have not been redrawn.

The simplified stride shape was checked for 576 combinations of odd kernel
sizes, strides and odd/even spatial dimensions against the documented output
formula. The retained stride diagram's five values and three sampled values,
and the max-pooling diagram's four values, were recomputed from their displayed
inputs. These are arithmetic checks, not execution of a PyTorch model.

The prose received the humanizer pass against the preceding section's voice,
including first-person plural for instructional actions. Final rendering of the
CNN specification and blocks slides confirms that both captions clear the
footer at the existing body font size.

References checked for the corrections:
[PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
[PyTorch MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html),
and Richard Zhang's [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486).
The existing stride figure retains its [CS231n source](https://cs231n.github.io/convolutional-networks/).

This is a rendered candidate for instructor review. Simone requested a light
pass; acceptance of the revised wording and layout remains open. No tracked
download PDF has been exported. Continue with **Other notable types of
convolutions** after this section's review.
