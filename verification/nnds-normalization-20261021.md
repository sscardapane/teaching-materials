# NNDS normalization revision — 2026-10-21

Written by Codex (2026-10-21).

Simone requested implementing four recorded review points and sending screenshots:
show the BN smoothing claim, connect larger learning rates to the later argument,
clarify scale invariance and weight decay, and connect LN geometry to BN/LN
optimization. These are slide-specific edits to
`courses/nnds/Lecture_deeper_models.tex`; the existing visual system is retained.
The screenshots are for instructor review. No public PDF export is included.

## Dependencies and notation

| Explanation | Prerequisites already introduced | Notation and scope |
| --- | --- | --- |
| Smoothing evidence | Loss, gradients, gradient descent; preceding BN explanation | Full network parameters `theta`; `g = gradient L(theta)`; probes `theta - alpha g` |
| Scale cancellation | BN formula and its batch statistics | Positive scale `a`; training statistics; fixed gamma/beta; epsilon neglected |
| Gradient identities | `L(a w) = L(w)` | Data loss, chain rule in `a` and in `w`; pre-normalization weights used only through this path |
| Effective learning rate | Gradient orthogonality and inverse gradient scaling | `w = r u`, `r = norm(w)`, unit `u`; gradient descent without momentum |
| Weight decay | Effective learning rate and orthogonality | `lambda` is decay strength, `0 < eta lambda < 1`; L2 equivalence stated for gradient descent |
| BN/LN comparison | BN/LN axes, LN geometry, scale-invariant weight updates | BN per unit/filter; LN whole preceding affine map, bias included; CNN BN vector length `BHW` |

The LN optimization argument remains valid at the scale of a whole affine block.
It does not imply independent invariance of each incoming unit weight vector.
BN geometry concerns values across examples; LN geometry concerns a single
example's features. The smoothing evidence and the scale-invariance calculation
are complementary explanations, rather than a universal proof of learning-rate
stability.

## Coverage of the original source

The baseline source SHA256 was
`e692545fcf903eca0557a01bb78fe2b5ac8bf0be6aa92b7a867d97df082d2ec5`.
There were 21 explicit frames (24 PDF pages, including subsection dividers).
There are now 25 explicit frames (28 PDF pages). No original frame, citation,
figure, experiment result or derivation was removed.

| Original frame | Preservation status | Location or change |
| --- | --- | --- |
| Title page | leave | Unchanged |
| ImageNet and its challenge | leave | Unchanged |
| Going deeper on ImageNet | leave | Unchanged |
| From LeNet to AlexNet | leave | Unchanged |
| Can we just stack more layers? | leave | Unchanged |
| The rest of the lecture | leave | Unchanged |
| Normalizing the inputs | leave | Unchanged |
| Batch normalization | leave | Unchanged |
| Batch normalization in CNNs | leave | Unchanged |
| Batch normalization at inference | leave | Unchanged |
| How the running average works | leave | Unchanged |
| Fusing batch normalization at inference | leave | Unchanged |
| Layer normalization | leave | Unchanged |
| Why normalize across features? | leave | Unchanged |
| What layer normalization does to a vector | light edit | Explicit epsilon approximation and local paragraph spacing; example and native geometry retained |
| Layer normalization for images and for LLMs | leave | Unchanged |
| RMSNorm: dropping the mean | leave | Unchanged |
| Why does normalization help? | leave | Preserved, including all three references and the uncertainty statement; followed by visual evidence |
| Normalization allows larger learning rates | light edit | Existing experiment, curves and all numbers retained; added bridge from nominal learning rate to weight norm |
| Normalized layers ignore the scale of their weights | substantial edit | Expanded scale cancellation; both gradient identities and the original ray figure preserved on the next slide |
| Weight decay as a learning rate | substantial edit | Original inverse-square effective rate is derived on the preceding added slide; Pythagorean norm growth retained here, followed by explicit decay update and its norm recurrence |

## Added frames

- **A more predictive gradient:** original Santurkar et al. Figure 4(a,b),
  with both measurement definitions and the interpretation of the shaded bands.
- **Scale invariance determines the gradient:** separate chain-rule calculations
  for gradient orthogonality and inverse scaling, plus the original ray figure.
- **The weight norm sets the effective learning rate:** exact normalized update
  and its small-step interpretation, deriving `eta_eff = eta / r^2`.
- **BN and LN: the same operation on different vectors:** shared shift/scale
  invariance; representation geometry versus per-unit or blockwise optimization.

## Figure provenance

`courses/nnds/assets/images/deeper/bn_smoothing.pdf` is a vector extraction of
Figure 4(a,b), PDF page 5, of Santurkar et al., *How Does Batch Normalization Help
Optimization?* (NeurIPS 2018): <https://arxiv.org/pdf/1805.11604v5>.
It is not a local reproduction or a redrawn curve. The crop retains both original
panels, including all axes, legends and panel labels. The adjacent panel (c) and
the paper's caption are outside the selected figure region.

`extract_bn_smoothing.py` records the source URL, crop coordinates and expected
source SHA256 `41d5195d959a8c37f3f23df34023032b2a0361887495cdb558a2402f60437d1e`.
Run it with the repository's Python environment containing `pypdf`.

The optimization derivation also uses van Laarhoven (2017),
<https://arxiv.org/abs/1706.05350>, with a direct chain-rule derivation through
the normalization statistics. The comparison uses Ba et al. (2016), Section 5.1
and Table 1, <https://arxiv.org/abs/1607.06450>.

## Verification

- Built with the repository LuaLaTeX build script: 28 PDF pages.
- The required full NNDS build also passed for all six registered decks.
- Rendered and inspected all eight changed or added full slides at 1600 pixels
  wide, including title, body, plots, equations, captions and footer. Corrected
  overflow and a diagram label placement before delivery.
- No overfull horizontal or vertical boxes on the revised slides. The two
  existing vertical warnings remain on the unchanged title page (29.87 pt)
  and Batch normalization (4.92 pt).
- Compared every original frame against the pre-edit source; 17 are verbatim,
  four are edited as mapped above, and every original citation remains.
- Independently checked scale invariance, gradient orthogonality, inverse
  gradient scaling, the exact direction update and the norm update with decay
  using finite differences of a normalized linear model. Largest residual was
  below `4e-10`.
- Instructor acceptance of these candidates remains pending. Residual connections
  and the rest of the lecture are outside this revision.
