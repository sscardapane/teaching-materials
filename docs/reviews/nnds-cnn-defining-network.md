# CNN defining-the-network review

> [!note] Written by Codex (2026-10-14)

## Scope

Simone resumed the CNN refresh and said the next section could remain more or
less similar. This candidate keeps the nine slides of **Defining the network**
in their original order, with all four original figures. The authoritative
source is `courses/nnds/Lecture_CNNs.tex`. It follows the opening and
convolutional-layer checkpoints recorded in this directory.

The initial candidate in `26a6abb` unnecessarily rewrote unchanged prose. Simone
rejected that approach and pointed out the inconsistent block definition. This
revision starts again from `7bc132e`, preserving unaffected source text and
formatting. It applies only the corrections listed below.

The edits are local to this subsection. Its slides remain on PDF pages 18 to 26
(printed numbers 15 to 23). The complete deck still has 26 explicit frames and
30 PDF pages, including dividers. The accepted opening, the preceding
convolutional-layer candidate and everything from **Other notable types of
convolutions** onward are unchanged.

## Original-slide coverage

| Original slide | Status | Result |
| --- | --- | --- |
| Reducing dimensionality | light edit | Compose the previously defined convolutional layers f_i, each including its activation. Keep the original prose except the padding qualification and corrected block definition. |
| Stride | light edit | Use r for stride, retaining s for kernel width. For stride 2, use P_k(2i,2j), W^T, bias and the complete channel slice. Keep the original example structure; correct output-shape rounding and replace TensorFlow with nn.Conv2d. |
| Convolution with a larger stride | leave | Preserve the entire source frame, caption, attribution and figure scale verbatim. |
| Max-pooling | leave | Preserve the entire source frame and wording verbatim. |
| Visualization of max-pooling | light edit | Preserve the original caption and figure scale verbatim. Reduce only local top spacing to clear the restored caption. |
| Complete architecture of a CNN (1), previously titled only (1) | light edit | Restore the title and qualify exact invariance. Preserve the original opening, bullets and model-selection paragraph. |
| Complete architecture of a CNN (2) | light edit | Restore the original equation wording and colors. Complete the channel slice and reduce array spacing to remove overflow. |
| Example of a simple CNN specification | light edit | Correct the claim that repeated downsampling is required before classification. Keep the original figure; reduce its displayed width and local spacing so the caption clears the footer. |
| Designing in blocks | light edit | Restore the original caption verbatim. Keep the original figure and scale, changing only local top spacing. |

No slides, definitions or figures are removed, added or moved. All four figures
retain their source files and provenance. No shared style, notebook, other deck,
HTML page or published PDF changes form part of this candidate.

## Dependencies and notation

| Slide | Prerequisites |
| --- | --- |
| Reducing dimensionality | light edit | Compose the previously defined convolutional layers f_i, each including its activation. Keep the original prose except the padding qualification and corrected block definition. |
| Stride | light edit | Use r for stride, retaining s for kernel width. For stride 2, use P_k(2i,2j), W^T, bias and the complete channel slice. Keep the original example structure; correct output-shape rounding and replace TensorFlow with nn.Conv2d. |
| Max-pooling and its figure | Image grid and stride. |
| Architecture overview | Convolution, downsampling, pooling and earlier translation equivariance. |
| Architecture equation and example | Flattening and global pooling from the preceding slide; softmax and the classifier from linear models. |
| Designing in blocks | light edit | Restore the original caption verbatim. Keep the original figure and scale, changing only local top spacing. |

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
including first-person plural for instructional actions. The restored prose is not subjected to a new style pass. Final rendering confirms
that the max-pooling visualization, CNN specification and blocks captions clear
the footer at the existing body font size. Source checks compare untouched
frames and captions byte for byte with `7bc132e`.

References checked for the corrections:
[PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
[PyTorch MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html),
and Richard Zhang's [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486).
The existing stride figure retains its [CS231n source](https://cs231n.github.io/convolutional-networks/).

This is a rendered candidate for instructor review. Simone requested a light
pass; acceptance of the revised wording and layout remains open. No tracked
download PDF has been exported. Continue with **Other notable types of
convolutions** after this section's review.
