# Experiments for "Building deeper models"

Every plot in the NNDS lecture on deeper models comes from the scripts in this
folder. The raw runs are in `results/`, so you can rebuild any table without
training anything, or retrain any part and compare.

All networks are the CIFAR-10 ResNet of He et al. (2016, Sec. 4.2) and its
variants (plain, plain + BN, residual, residual with zero-initialized branches,
residual + BN), trained for 30 epochs with SGD (Nesterov momentum), weight
decay 5e-4, warmup then cosine schedule, batch 128, one seed. `depth_sweep.py`
explains the setup in its header.

## Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy pyarrow pillow matplotlib
```

CIFAR-10 is downloaded on the first run to `~/.cache/cifar10/hf`. The scripts
pick CUDA, then an Apple GPU, then the CPU. The times below are from an Apple
M-series GPU; CPU training is much slower.

For a one-minute smoke test (1 epoch, depths 8 and 20, written to a temporary
directory):

```bash
.venv/bin/python courses/nnds/assets/images/deeper/depth_sweep.py --quick
```

## From raw runs to slide tables (no training)

```bash
cd courses/nnds/assets/images/deeper
../../../../../.venv/bin/python opener_table.py
../../../../../.venv/bin/python residual_table.py
../../../../../.venv/bin/python lr_table.py
../../../../../.venv/bin/python smoothing_table.py
../../../../../.venv/bin/python regularization_table.py
```

Each script overwrites the CSV that its slides plot. Then rebuild the deck with
`python3 scripts/build.py nnds` from the repository root.

## Retraining, slide by slide

Commands run from the repository root; `S` stands for
`.venv/bin/python courses/nnds/assets/images/deeper` and `R` for
`courses/nnds/assets/images/deeper/results`. Finished runs are skipped, so
each command can be stopped and resumed. To retrain from scratch, move the
existing JSON files away first.

| Slides | Command | Runs | Time | Table |
|---|---|---|---|---|
| "Can we just stack more layers?", "Deeper should not be worse", "Back to the opening experiment", "Residuals without normalization" | `$S/depth_sweep.py --variants plain --init he default --depths 8 20 32 44 56 --lr 0.005 0.02 0.05 --out $R/opener`<br>`$S/depth_sweep.py --variants res res-zero --depths 8 20 32 44 56 --lr 0.005 0.02 0.05 --out $R/opener`<br>`$S/depth_sweep.py --variants plain-bn res-bn --depths 8 20 32 44 56 --lr 0.02 0.05 0.1 --out $R/opener` | 90 | 7.5 h | `opener_table.py` → `opener.csv` |
| "Differentiating residual connections" (gradient at initialization), "Removing one block" (lesion test) | the 56-layer runs above, plus `$S/depth_sweep.py` with its defaults (depths 8, 20, 56 at learning rate 0.05; some lesion tests come from these runs) | 15 | 1 h | `residual_table.py` → `grad_init.csv`, `lesion.csv` |
| "Normalization allows larger learning rates" | `$S/depth_sweep.py --variants plain plain-bn --depths 20 --lr 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 --out $R/lr_sweep` | 18 | 1 h | `lr_table.py` → `lr.csv` |
| "A more predictive gradient" | `$S/smoothing_probe.py`<br>`$S/smoothing_probe.py --variants plain-bn --lr 0.1` | 3 | 0.5 h | `smoothing_table.py` → `smoothing.csv` |
| "Regularization at work" | `$S/depth_sweep.py --variants res-bn --depths 20 --out $R/regularization`, then the same with `--wd 0`, with `--no-augment`, and with `--wd 0 --no-augment` | 4 | 1.5 h | `regularization_table.py` → `regularization.csv`, `regularization_curves.csv` |

For each depth and variant, the tables keep the learning rate with the lowest
final training loss, and count a diverged run as chance (10% accuracy).

## What to expect

GPU training is not bit-for-bit repeatable, and every point is a single seed.
A rerun gives the same trends, with accuracies that can move by a few tenths
of a point. Adding seeds (`--seeds 0 1 2`) and averaging is a good exercise.

## Other scripts

- `plot_depth_sweep.py`: one overview page of a results folder, for exploring
  runs rather than for the slides.
- `mixing_figure.py`, `mnist_strip.py`: the MixUp/CutMix and MNIST image strips.
- `extract_bn_smoothing.py`: crops a figure from Santurkar et al. (2018); kept
  for provenance, the slide now uses our own measurement.
