# CNN convolutional-layers review

> [!note] Written by Codex (2026-10-13)

## Scope and decisions

This continues the accepted opening in [CNN opening review](nnds-cnn-opening.md).
The authoritative source is `courses/nnds/Lecture_CNNs.tex`. Simone approved
shortening **Convolutional layers** and then requested a rendered preview.
The section now has eight content slides, down from eighteen. They occupy PDF
pages 9 to 16, with printed slide numbers 7 to 14. The complete deck has 26
explicit frames and 30 PDF pages, including dividers.

Simone explicitly asked to remove **Filter operation 1/3**, **2/3** and **3/3**:
"they are very old and I can do that at the blackboard, it's also redundant now
that we have the previous figure." This authorizes his own walkthrough of this
operation. It does not change the course-wide rule against proposing live board
derivations.

He first said he liked the book's patch definition because it supports
`W^T vect(P_ij)`, then clarified: "I wanted to keep that equation roughly
similar, even if we don't use \"patches\"." The candidate therefore keeps the
familiar affine-layer equation, including `W^T`, vectorization, bias and the
following activation. It introduces the local tensor briefly, with the book's
existing `P_k(i,j)` notation and figure, without the pixel-distance definition.

The implementation slide connects all local affine maps to unfolding and one
matrix product, followed by a short PyTorch example. Toeplitz structure remains
in the four-element example. Stride and its sampling implications belong to the
next subsection. Memory-stride views remain a possible spoken or lab aside.

After reviewing the first render, Simone requested a separate zero-padding slide
using the running example, including the amount needed to preserve spatial size.
It comes before the Toeplitz matrix so its boundary zeros have already been
introduced. For the odd, contiguous kernels evaluated at every position in this
section, padding is `(s-1)/2` on each side.

On the next review, Simone asked to show this padding in the running example
itself. Its product now uses the explicit six-element padded input and a
`4 x 6` Toeplitz matrix. The operation and the three learned coefficients are
unchanged. The sixteen-coefficient comparison refers to the earlier dense layer
on the original four-element input.

He also asked for consistent first-person plural wording. Instructional actions
now use "we take", "we apply", "we add" and similar constructions. Definitions
remain direct statements. The complete eight-slide section received a wording
review for grammar, notation and precision, including the boundary condition on
the receptive-field width formula. The voice convention is recorded in
`docs/NNDS_AUTHORING.md` for subsequent revisions.

He also requested `+b` with implicit broadcasting, matching the MLP convention.
The implementation slide now uses that notation. Its explanation follows his
suggested wording: `nn.Conv2d` uses a hardware-specific implementation, so our
code does not construct `U(X)` explicitly. The details are left to the lab; the
slide does not claim that every backend internally avoids materializing it.

These are changes to this finite set of slides. No shared template changes or
HTML conversion form part of this candidate. The accepted opening is identical.
Everything from **Defining the network** to the end remains identical to the
previous source. Simone asked to check those slides only when their turn comes;
notation and implementation references there remain for that later review.

## Original-slide coverage

Each original slide has one preservation status.

| Original slide | Status | Result |
| --- | --- | --- |
| What we want | remove | The opening and local-neighborhood definition provide the motivation. |
| Distance between pixels | substantial edit | Local neighborhoods retains the radius, kernel size and local-tensor shape; omit the distance metric. |
| Visualizing a patch | move | Embed the original book figure in Local neighborhoods. |
| Locality | substantial edit | Retain the locality definition on Local neighborhoods and the affine-map form on Convolutional layer; omit position-specific weights. |
| Running example (unshared local weights) | remove | Keep the final shared-weight example instead. |
| Zero-padding | substantial edit | Keep a separate slide using the padded four-element sequence; explain the amount that preserves size. |
| Image padding | remove | The boundary example covers padding. |
| Translational equivariance | substantial edit | Define translation equivariance by shifting input and output; state the finite-boundary qualification. |
| Convolutional layer (equations) | substantial edit | Keep one affine-plus-activation equation with transposed weights, correct shapes and the bias-inclusive parameter count. |
| Running example (shared weights) | substantial edit | Show the padded six-element input and its rectangular Toeplitz matrix explicitly; retain the coefficient comparison and omit bias and activation. |
| Equivalent formulation | remove | The vectorized-neighborhood equation defines the operation; omit the duplicate indexing/FIR formulation. |
| Terminology time! | substantial edit | Introduce kernel size, filters and activation maps where they are used. |
| Filter operation 1/3 | remove | Explicitly requested. |
| Filter operation 2/3 | remove | Explicitly requested. |
| Filter operation 3/3 | remove | Explicitly requested. |
| Convolutional layer (multiple-filter figure) | light edit | Preserve the figure; explain filter columns and output-channel slices. |
| Receptive field | substantial edit | Retain the definition and add the width formula and the 3/5/7 example for contiguous kernels evaluated at every pixel. |
| Implementation | substantial edit | Replace TensorFlow with PyTorch; connect unfolding to matrix multiplication, use broadcast bias and explain hardware specialization in plain language. |

## Dependencies and notation

| Slide | Prerequisites |
| --- | --- |
| Local neighborhoods | Image axes and channels from the accepted opening. |
| Convolutional layer | Local tensor, vectorization, affine maps and pointwise activations. |
| Filters and output channels | Shared weight matrix and output channels. |
| Zero-padding | The four-element sequence and kernel size. Introduces boundary handling before the Toeplitz example. |
| Running example | The four-element sequence, kernel size, shared weights and padding. |
| Translation equivariance | Shared weights and padding. Introduces the shift operator before using it. |
| Receptive field | Local layers and composition from the MLP unit. |
| Implementation | Vectorized neighborhoods, shared affine maps and padding. |

- `s=2k+1` is the kernel width, and `P_k(i,j)` has shape `(s,s,c)`.
- `W` has shape `(s^2 c,c')`. Its columns are filters, so `W^T` maps a flattened
  neighborhood into `c'` output channels. Bias has shape `(c')`.
- `H` is the output after the pointwise activation. The four-element matrix
  example explicitly omits bias and activation.
- `U(X)` contains neighborhoods as columns. With zero padding preserving the
  spatial size, it has shape `(s^2 c,hw)`. `Z` denotes pre-activation outputs;
  the implementation equation uses their matrix form, `\widetilde Z`.
- The bias is implicitly broadcast across the spatial positions, as in the MLP
  slides. The PyTorch code adds the batch axis and uses `(B,C,H,W)` ordering,
  shown in its comment.

## Verification and review

`python3 scripts/build.py nnds` builds all five NNDS decks. Every changed slide
has been rendered and inspected in full. The neighborhood figure's height and
implementation spacing were adjusted after the first render. The zero-padding and receptive-field slides use local paragraph spacing, and
zero-padding also uses local equation spacing, to fit at the existing font size.
The final eight slides have no overfull box warnings. The title and later unreviewed slides
retain their existing warnings.

The displayed PyTorch example runs and returns shape `(1,100,64,64)`. Its
`Unfold` plus matrix-product equivalent agrees with `Conv2d` within floating-point
tolerance (maximum absolute error `7.15e-7`). The rectangular Toeplitz matrix
acting on the explicit padded input agrees with both the previous reduced
matrix and zero-padded `Conv1d`. Additional checks confirm that padding one cell for
a size-3 kernel and two for a size-5 kernel preserve both sequence length and
2D spatial dimensions; without padding, the four-element size-3 example has
two outputs. Source checks verify that the accepted opening is
unchanged and that everything after this subsection is unchanged.
All ten imported figure exports still match their recorded source hashes; removed
figures remain in the repository for provenance.

The slide prose and this record received the humanizer pass against the existing
course voice. The implementation sources appear as links on the slide:
[PyTorch Unfold](https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html)
and [NVIDIA convolution implementations](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html).

This is a review candidate. Simone has approved its direction and removals;
acceptance of the rendered wording and layout is still pending. The source and
review record form a Git checkpoint. No tracked download PDF is exported.
