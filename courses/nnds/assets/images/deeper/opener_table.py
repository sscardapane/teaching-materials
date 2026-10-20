"""Table for the opening experiment and the reruns that close each section.

Reads results/opener/*.json (depth_sweep.py runs with several learning rates)
and keeps, for each depth, variant and initialization, the learning rate with
the lowest final training loss. A diverged run counts as chance (loss ln 10,
accuracy 10%). Writes opener.csv, which the slides plot with pgfplots: one
column group per variant and initialization, named e.g. plain_he_loss or
plainbn_he_acc. Missing runs are written as nan.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANCE_LOSS, CHANCE_ACC = math.log(10), 0.1

best = {}
for f in sorted((HERE / "results" / "opener").glob("*.json")):
    r = json.loads(f.read_text())
    h = r["history"]
    if r["diverged"] or not h["train_loss"]:
        loss, acc = CHANCE_LOSS, CHANCE_ACC
    else:
        loss, acc = h["train_loss"][-1], h["test_acc"][-1]
    key = (r["depth"], r["variant"].replace("-", "") + "_" + r["init"])
    if key not in best or loss < best[key][0]:
        best[key] = (loss, acc, r["lr"])

depths = sorted({d for d, _ in best})
groups = sorted({g for _, g in best})
rows = [",".join(["depth"] + [f"{g}_{m}" for g in groups for m in ("loss", "acc", "lr")])]
for d in depths:
    cells = [str(d)]
    for g in groups:
        loss, acc, lr = best.get((d, g), (math.nan, math.nan, math.nan))
        cells += [f"{loss:.4f}", f"{100 * acc:.2f}", f"{lr:g}"]
    rows.append(",".join(cells))
(HERE / "opener.csv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
