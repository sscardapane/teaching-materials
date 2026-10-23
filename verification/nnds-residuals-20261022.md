# NNDS deeper models: residual connections (2026-10-22)

Written by Claude. Source deck: the 2024 unit
`NNDS_2024/8_Building_Deep_Networks/Lecture_8_building_deep_networks.tex`
(Dropbox), subsection "Residual connections" (10 frames). Target:
`courses/nnds/Lecture_deeper_models.tex`, section "Residual connections".

## Coverage of the 2024 frames

| 2024 frame | Status | Where now |
|---|---|---|
| Adding layers (He et al. 2016, Fig. 1) | substantial edit | "Deeper should not be worse": same argument; our plain+BN training loss against depth replaces the paper figure |
| Residual connections | light edit | "Residual connections": same definition and rescaling remark; adds the existing residual TikZ figure. The De and Smith (2020) remark moves to "Residuals without normalization" |
| Differentiating residual connections | substantial edit | same title: same Jacobian and VJP; adds our gradient-at-initialization measurement |
| Residual connections in practice (D2L figure) | merged | "The residual block in practice": own TikZ figure; keeps "ReLU after the sum" |
| Residual connections with rescaling (D2L figure) | merged | same slide, right-hand block |
| Residual connections with rescaling (full ResNet, D2L) | merged | same slide, last paragraph (stages, 6n+2 layers) |
| Other families of neural networks (ResNet/ConvNeXt figure) | substantial edit, moved | "The best CNNs today", at the end of the residual section: same figure and purpose (the best a CNN achieves today, per Simone); adds ConvNeXt V2 (88.9%, public data) and NFNets at scale (90.4%), which links back to residuals without normalization |
| Hubs | removed | approved in the plan of 2026-10-22; fits better where pretrained models appear |
| ResNets as ensembles | light edit | same title and figure; adds our lesion test on the next slide |
| Neural ODEs | light edit | same content; the limit is now dh(t)/dt = f(h(t), t) (2024: ∂h_t/∂t = f(x, t)); adds Chen et al. (2018) |
| Reading material (Chapter 9) | deferred | belongs at the end of the lecture, after regularization |

All 2024 citations remain (He et al. 2016; De and Smith 2020; Liu et al. 2022;
Veit et al. 2016). New: Yang et al. (2019), Goyal et al. (2017), Zhang et al.
(2019), Bachlechner et al. (2021), Chen et al. (2018).

## New slides

- "Residuals without normalization": variance doubling with He initialization;
  BN in the branch (De and Smith 2020) and starting from the identity
  (zero-initialized last layer, SkipInit/ReZero, Fixup).
- "Removing one block": lesion test on the trained 56-layer networks.
- "Back to the opening experiment": the opening figure with plain+BN and
  residual+BN added; bridge to regularization.

## Experiments

All from `assets/images/deeper/depth_sweep.py`, results in
`results/opener/` (He initialization, 30 epochs, SGD with Nesterov momentum,
weight decay 5e-4, warmup then cosine, one seed). For each point the learning
rate with the lowest final training loss is kept (opener_table.py,
residual_table.py). Residual+BN: 0.02, 0.05, 0.1; residual without BN and
zero-initialized branches: 0.005, 0.02, 0.05.

- `grad_init.csv`: RMS gradient of the first convolution in each block, at
  initialization, 56 layers. It does not depend on the learning rate (same
  seed, same initialization).
- `lesion.csv`: test accuracy after removing each block whose input and output
  shapes agree (the two downsampling blocks are skipped).

# Regularization and closing (2026-10-22)

Simone asked to finish the lecture in the same session. Sources: the 2024
sections "Data strategies", "Optimization strategies" and "New layers /
Dropout regularization", plus the overfitting frames of the 2024
introduction. Basic augmentation is already in the CNN deck ("Data
augmentation"), the L2 penalty and AdamW in the shared MLP slides ("Weight
decay"); this section builds on both instead of repeating them.

## Coverage of the 2024 frames

| 2024 frame | Status | Where now |
|---|---|---|
| What about overfitting? | light edit | "Can deep networks overfit?" (definition card) |
| Visualization of overfitting in deep networks | merged | same slide; Zhang et al. Fig. 1a cropped (panel a only) |
| The double descent phenomenon | light edit | same title; figure legend cropped and explained in the text |
| Finding a stopping point | merged | "Early stopping" (first paragraph) |
| Early stopping | light edit | same title; adds "keep the best checkpoint" and one line on LLMs |
| Improving training with regularization | merged | "Weight decay, again" (penalty card) |
| Weight decay | merged | "Weight decay, again": now connects the MLP penalty and the effective learning rate of the normalization section; AdamW citation updated to the ICLR 2019 paper |
| Other forms of regularization | light edit | same title; Scardapane et al. (2017) kept, group sparsity in one sentence |
| Graphical depiction of LASSO | merged | same slide, right column |
| On weight reparameterization | light edit | same title; the C/2 factor makes the equivalence exact, with the one-line reason |
| Data augmentation | covered elsewhere | CNN deck, "Data augmentation" |
| Visualizing image augmentation | covered elsewhere | CNN deck figure |
| Other types of data augmentation (MixUp) | merged | "Mixing examples: MixUp and CutMix", with our CIFAR-10 figure (mixing_figure.py) |
| CutMix | merged | same slide; lambda is now the fraction of the image kept from x1 (the 2024 slide sampled it independently of the patch) |
| Pipelines for data augmentation (RandAugment code) | merged | "Augmentation pipelines": the code becomes one paragraph |
| Sampling data augmentation strategies | merged | same slide, book figure |
| Is augmentation part of the model? | substantial edit | "Augmentation in code": rewritten for PyTorch (2024 code was tf.keras) |
| Training and inference | merged | same slide, orange card linking to model.train()/model.eval() |
| Data augmentation revisited | merged | "From noisy inputs to noisy layers" |
| Visualization of dropout | merged | same slide, book figure |
| Dropout regularization | light edit | same title; H ~ (B, D) in the course convention |
| Montecarlo dropout | light edit | "Monte Carlo dropout" |
| Dropout at inference time | light edit | same title; adds nn.Dropout and model.eval() |
| Variants of dropout | light edit | same title; adds stochastic depth, linked to the lesion slide |
| Early and late dropout | substantial edit | "Dropout today": same figure, plus current practice (PaLM pretrained without dropout; DeiT uses stochastic depth) |
| Reading material | light edit | "Summary and reading material" |

Not carried over from the 2024 introduction, for Simone to confirm (the
opening section of 2026-10-20 replaced the VGG/CIFAR experiment with our
depth sweep): "How many layers are enough?", "Visualization of VGG-11",
"Experimenting on CIFAR-10", "Results", "Comments on the results", "Families
of deep networks", "Unexplained phenomena", "A real-world scaling law",
"Summarizing". Scaling laws and emergent abilities fit the LLM part of the
course better.

## New

- "Regularization at work": the 20-layer residual network with and without
  augmentation and weight decay (depth_sweep.py --no-augment / --wd 0,
  results/regularization/, regularization_table.py).
