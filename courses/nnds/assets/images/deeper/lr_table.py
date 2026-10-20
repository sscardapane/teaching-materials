"""Table for the "larger learning rates" figure in the normalization section.

Reads results/lr_sweep/*.json (depth_sweep.py at depth 20, He initialization,
plain and plain-bn variants, learning rates from 0.002 to 1) and writes
lr.csv: final training loss and test accuracy per learning rate and variant.
A diverged run counts as chance (loss ln 10, accuracy 10%) and is flagged in
the *_div column, so the slide can mark it. Missing runs are written as nan.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANCE_LOSS, CHANCE_ACC = math.log(10), 0.1
VARIANTS = ["plain", "plain-bn"]

runs = {}
for f in sorted((HERE / "results" / "lr_sweep").glob("*.json")):
    r = json.loads(f.read_text())
    h = r["history"]
    if r["diverged"] or not h["train_loss"]:
        runs[(r["lr"], r["variant"])] = (CHANCE_LOSS, CHANCE_ACC, 1)
    else:
        runs[(r["lr"], r["variant"])] = (h["train_loss"][-1], h["test_acc"][-1], 0)

lrs = sorted({lr for lr, _ in runs}) or [0.002, 1.0]
header = ["lr"] + [f"{v.replace('-', '')}_{m}" for v in VARIANTS for m in ("loss", "acc", "div")]
rows = [",".join(header)]
for lr in lrs:
    cells = [f"{lr:g}"]
    for v in VARIANTS:
        loss, acc, div = runs.get((lr, v), (math.nan, math.nan, 0))
        cells += [f"{loss:.4f}", f"{100 * acc:.2f}", str(div)]
    rows.append(",".join(cells))
(HERE / "lr.csv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
