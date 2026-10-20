"""Overview of the depth sweep results, for choosing the slide figures.

Reads results/*.json written by depth_sweep.py and draws one review page:
training loss per epoch (BN variants, the degradation problem), final training
loss against depth for every variant, gradient norms at initialization along
the deepest networks, and the lesion test. The slide figures themselves will
be TikZ with the measured values embedded, as in lecture_4/init/.

    .venv/bin/python courses/nnds/assets/images/deeper/plot_depth_sweep.py [results_dir] [out.png]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"plain": "#c0392b", "plain-bn": "#e67e22", "res": "#7f8c8d",
          "res-zero": "#27ae60", "res-bn": "#1f6fa8"}
ORDER = ["plain", "plain-bn", "res", "res-zero", "res-bn"]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "overview.png"
    runs = defaultdict(list)
    for f in sorted(src.glob("*.json")):
        r = json.loads(f.read_text())
        runs[(r["variant"], r["depth"])].append(r)
    depths = sorted({d for _, d in runs})
    deepest = depths[-1]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    plt.rcParams.update({"font.size": 11})

    a = ax[0, 0]
    for variant, style in (("plain-bn", "--"), ("res-bn", "-")):
        for d, lw in zip(depths, np.linspace(1, 3, len(depths))):
            for r in runs.get((variant, d), []):
                y = r["history"]["train_loss_steps"]
                if y:
                    a.plot(np.arange(len(y)) / 10, y, style, lw=lw,
                           color=COLORS[variant], label=f"{variant}-{d}")
    a.set(title="Training loss, with batch norm", xlabel="epoch", ylabel="loss")
    a.legend(fontsize=9)

    a = ax[0, 1]
    for variant in ORDER:
        xs, ys, div = [], [], []
        for d in depths:
            for r in runs.get((variant, d), []):
                if r["diverged"]:
                    div.append(d)
                else:
                    xs.append(d)
                    ys.append(r["history"]["train_loss"][-1])
        if xs:
            a.plot(xs, ys, "o-", color=COLORS[variant], label=variant)
        if div:
            a.plot(div, [a.get_ylim()[1]] * len(div), "x", ms=10, color=COLORS[variant])
    a.set(title="Final training loss vs depth (x = diverged)", xlabel="depth",
          ylabel="loss", xticks=depths)
    a.legend(fontsize=9)

    a = ax[1, 0]
    for variant in ORDER:
        for r in runs.get((variant, deepest), []):
            a.semilogy(np.arange(1, len(r["grad_norm_init"]) + 1), r["grad_norm_init"],
                       "o-", ms=3, color=COLORS[variant], label=variant)
    a.set(title=f"Gradient RMS at initialization, depth {deepest}",
          xlabel="block (input side on the left)", ylabel="RMS gradient")
    a.legend(fontsize=9)

    a = ax[1, 1]
    width = 0.4
    for k, variant in enumerate(("plain-bn", "res-bn")):
        for r in runs.get((variant, deepest), []):
            if "lesion_acc" not in r:
                continue
            idx = [i for i, v in enumerate(r["lesion_acc"]) if v is not None]
            vals = [r["lesion_acc"][i] for i in idx]
            a.bar(np.arange(len(idx)) + k * width, vals, width, color=COLORS[variant],
                  label=f"{variant} (intact {r['baseline_acc']:.2f})")
    a.axhline(0.1, color="k", lw=0.8, ls=":")
    a.set(title=f"Test accuracy after removing one block, depth {deepest}",
          xlabel="removed block (identity-shaped blocks only)", ylabel="accuracy",
          ylim=(0, 1))
    a.legend(fontsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for (variant, d), rs in sorted(runs.items(), key=lambda kv: (kv[0][1], ORDER.index(kv[0][0]))):
        for r in rs:
            h = r["history"]
            last = (f"train {h['train_loss'][-1]:.3f} test acc {h['test_acc'][-1]:.3f}"
                    if h["train_loss"] else "no epochs")
            print(f"{variant:>8}-{d:<3} params {r['params']:>7} "
                  f"{'DIVERGED ' if r['diverged'] else ''}{last} ({r['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
