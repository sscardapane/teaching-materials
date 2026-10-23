"""Tables for the slide that closes the regularization section.

Reads results/regularization/*.json: the 20-layer residual network with BN
(depth_sweep.py, eta = 0.05), trained with and without data augmentation
(random shifts and flips) and weight decay (5e-4). Writes:

* regularization_curves.csv: training and test loss per epoch, columns
  <config>_train and <config>_test;
* regularization.csv: final training loss, test loss and test accuracy (%)
  for each configuration.

Configurations: both (augmentation and weight decay), aug (augmentation
only), wd (weight decay only), none.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {(True, True): "both", (True, False): "aug", (False, True): "wd", (False, False): "none"}

runs = {}
for f in sorted((HERE / "results" / "regularization").glob("res-bn-20*.json")):
    r = json.loads(f.read_text())
    runs[NAMES[(r.get("augment", True), r.get("weight_decay", 5e-4) > 0)]] = r
configs = [c for c in ("both", "aug", "wd", "none") if c in runs]

epochs = runs[configs[0]]["history"]["epoch"]
rows = ["epoch," + ",".join(f"{c}_{k}" for c in configs for k in ("train", "test"))]
for i, e in enumerate(epochs):
    cells = [str(e)]
    for c in configs:
        h = runs[c]["history"]
        cells += [f"{h['train_loss'][i]:.4f}", f"{h['test_loss'][i]:.4f}"]
    rows.append(",".join(cells))
(HERE / "regularization_curves.csv").write_text("\n".join(rows) + "\n")

rows = ["config,train_loss,test_loss,test_acc"]
for c in configs:
    h = runs[c]["history"]
    rows.append(f"{c},{h['train_loss'][-1]:.4f},{h['test_loss'][-1]:.4f},{100 * h['test_acc'][-1]:.2f}")
(HERE / "regularization.csv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
