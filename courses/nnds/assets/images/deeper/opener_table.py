"""Table for the opening figure: plain networks of increasing depth.

Reads results/opener/*.json (depth_sweep.py with --variants plain, both
initializations and several learning rates) and keeps, for each depth and
initialization, the learning rate with the lowest final training loss. A
diverged run counts as chance (loss ln 10, accuracy 10%). Writes opener.csv,
which the slide plots with pgfplots; depths not yet run are written as nan.
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
    key = (r["depth"], r["init"])
    if key not in best or loss < best[key][0]:
        best[key] = (loss, acc, r["lr"])

depths = sorted({d for d, _ in best})
rows = ["depth,he_loss,he_acc,he_lr,default_loss,default_acc,default_lr"]
for d in depths:
    cells = [str(d)]
    for init in ("he", "default"):
        loss, acc, lr = best.get((d, init), (math.nan, math.nan, math.nan))
        cells += [f"{loss:.4f}", f"{100 * acc:.2f}", f"{lr:g}"]
    rows.append(",".join(cells))
(HERE / "opener.csv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
