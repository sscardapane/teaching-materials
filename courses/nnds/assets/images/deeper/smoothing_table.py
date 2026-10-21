"""Turn smoothing_probe.py runs into smoothing.csv for the slide.

One row per probe: the epoch, then for each run and step length alpha the
relative gradient change and the loss change, averaged over a window of
--window consecutive probes to make the curves readable.
Written by Claude (2026-10-21).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {"plain": "plain-20-lr0.02-s0", "bn": "plain-bn-20-lr0.02-s0", "bnhigh": "plain-bn-20-lr0.1-s0"}


def smooth(values, window):
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        chunk = v[max(0, i - window + 1):i + 1]
        if np.isfinite(chunk).any():
            out[i] = np.nanmean(chunk)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "smoothing")
    args = ap.parse_args()
    runs = {k: json.loads((args.results / f"{v}.json").read_text())
            for k, v in RUNS.items() if (args.results / f"{v}.json").exists()}
    first = next(iter(runs.values()))
    alphas, spe = first["alphas"], first["steps_per_epoch"]
    n = min(len(r["probes"]) for r in runs.values())
    columns = {"epoch": [p["step"] / spe for p in first["probes"][:n]]}
    for name, r in runs.items():
        probes = r["probes"][:n]
        for i, a in enumerate(alphas):
            columns[f"{name}_grad_{a:g}"] = smooth([p["grad_change"][i] for p in probes], args.window)
            columns[f"{name}_loss_{a:g}"] = smooth([p["loss_change"][i] for p in probes], args.window)
    with (HERE / "smoothing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in zip(*columns.values()):
            w.writerow(["nan" if not np.isfinite(x) else f"{x:.4g}" for x in row])
    print(HERE / "smoothing.csv")


if __name__ == "__main__":
    main()
