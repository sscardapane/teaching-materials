"""Tables for the residual-connections section of the deeper-models deck.

Reads the 56-layer runs in results/opener/ (depth_sweep.py with several
learning rates) and, for each variant, keeps the learning rate with the lowest
final training loss, as opener_table.py does. Writes:

* grad_init.csv: RMS of the gradient reaching the first convolution of each
  block at initialization (block 1 is closest to the input);
* lesion.csv: test accuracy after removing one block at a time from the
  trained network (Veit et al., 2016); blocks that change shape cannot be
  removed and are written as nan.

The slides plot both with pgfplots. Columns are named after the variants,
without dashes: plain, plainbn, res, reszero, resbn.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPTH = 56

best = {}
for f in sorted((HERE / "results" / "opener").glob(f"*-{DEPTH}-he-*.json")):
    r = json.loads(f.read_text())
    h = r["history"]
    loss = math.inf if r["diverged"] or not h["train_loss"] else h["train_loss"][-1]
    v = r["variant"].replace("-", "")
    if v not in best or loss < best[v][0]:
        best[v] = (loss, r)

# Some opener runs predate the lesion test. The 2026-10-19 runs in results/
# used the same setup (He initialization, eta = 0.05), so we take the lesion
# from there when the learning rate matches.
for v, (_, r) in best.items():
    old = HERE / "results" / f"{r['variant']}-{DEPTH}-s0.json"
    if "lesion_acc" not in r and old.exists():
        o = json.loads(old.read_text())
        if o["lr"] == r["lr"] and "lesion_acc" in o:
            r["lesion_acc"] = o["lesion_acc"]

variants = [v for v in ("plain", "plainbn", "res", "reszero", "resbn") if v in best]
for v in variants:
    print(f"{v}: lr {best[v][1]['lr']:g}, final training loss {best[v][0]:.4f}")


def write(name, key, scale=1.0):
    cols = {v: best[v][1].get(key) for v in variants}
    cols = {v: c for v, c in cols.items() if c}
    n = max(len(c) for c in cols.values())
    rows = ["block," + ",".join(cols)]
    for k in range(n):
        cells = [str(k + 1)]
        for c in cols.values():
            x = c[k] if k < len(c) else None
            cells.append("nan" if x is None else f"{scale * x:.6g}")
        rows.append(",".join(cells))
    (HERE / name).write_text("\n".join(rows) + "\n")
    print(name, "columns:", ", ".join(cols))


write("grad_init.csv", "grad_norm_init")
write("lesion.csv", "lesion_acc", scale=100)
