# ReLU geometry figures

Original editable TikZ figures for the NN-only MLP addition (October 1, 2026).
No external image assets are used. Colors and typography come from NN26 and
`nn-content.tex`.

- `regions.tex`: the exact activation partition of
  `f(x_1,x_2) = ReLU(x_1) + ReLU(x_2-x_1)` on the displayed square `[-2,2]^2`.
  The dashed boundaries are `x_1=0` and `x_2=x_1`. The four labels give the
  output and the activation pattern in the same order as the two ReLUs.
  Neighboring affine formulas agree on their common boundary. A zero
  preactivation contributes zero regardless of the gate convention there.
  Biases are zero in this example; the accompanying general equation includes them.
- `folding.tex`: exact vertices of `T`, `T composed with T`, and `T` composed
  three times, on `[0,1]`. Here
  `T(x) = 2 ReLU(x) - 4 ReLU(x-1/2) + 2 ReLU(x-1)`.
  The function is zero outside `[0,1]`; each composition maps `[0,1]` onto itself.
  Shading marks affine intervals, not separately trained experts.
  All three plots use identical axes. A block uses three ReLU units and an
  affine readout; merging consecutive affine maps realizes `k` blocks as `k`
  width-3 hidden layers followed by an affine output (weights can be shared).
  The construction is illustrative and does not claim minimal width.

The first slide's general piecewise-affine interpretation follows Montufar et al.
(2014), [On the Number of Linear Regions of Deep Neural Networks](https://arxiv.org/abs/1402.1869).
The second is an equivalent ReLU realization of the triangle-wave composition
in Section 3.3 of Telgarsky (2016),
[Benefits of depth in neural networks](https://proceedings.mlr.press/v49/telgarsky16.html).
This is a compact illustration of representation efficiency, not a statement
about the performance of trained models or a presentation of the full theorem.

`python3 verify_geometry.py` checks the actual plotted polygons against the
network, including boundaries, with rational arithmetic. It also checks all
vertices and interior sample points of the composed triangles through eight
compositions, and verifies their realization as a width-3 ReLU MLP.
