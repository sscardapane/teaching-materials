"""Gradient predictiveness during training, with and without batch normalization.

Our own version of the measurement in Santurkar et al. (2018, Fig. 4), on the
20-layer plain CNN of depth_sweep.py with its optimizer and schedule. Every
--every steps we freeze training, compute the gradient g on a fixed batch of
training images, and step along it by several lengths alpha:

* loss change:      L(theta - alpha g) - L(theta)
* gradient change:  ||grad L(theta - alpha g) - g|| / ||g||

then restore the parameters and continue. The step lengths are absolute (the
same for both networks), so they mean the same learning rate in both.

    .venv/bin/python courses/nnds/assets/images/deeper/smoothing_probe.py

Writes results/smoothing/<variant>-20-lr<lr>-s<seed>.json; smoothing_table.py
turns the runs into smoothing.csv for the slide. Written by Claude (2026-10-21).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from depth_sweep import HERE, Net, augment, load_cifar

ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.1]


def probe(model, x, y, alphas):
    params = [p for p in model.parameters()]
    saved = [p.detach().clone() for p in params]

    def loss_and_grad():
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        return loss.item(), torch.cat([p.grad.reshape(-1) for p in params])

    # BN layers stay in training mode (batch statistics), as during training;
    # the running averages are restored afterwards.
    buffers = [b.detach().clone() for b in model.buffers()]
    loss0, g = loss_and_grad()
    gnorm = g.norm().item()
    out = {"loss": loss0, "grad_norm": gnorm, "loss_change": [], "grad_change": []}
    for alpha in alphas:
        with torch.no_grad():
            offset = 0
            for p, s in zip(params, saved):
                n = p.numel()
                p.copy_(s - alpha * g[offset:offset + n].view_as(p))
                offset += n
        loss, g2 = loss_and_grad()
        out["loss_change"].append(loss - loss0 if math.isfinite(loss) else None)
        change = (g2 - g).norm().item() / gnorm
        out["grad_change"].append(change if math.isfinite(change) else None)
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s)
        for b, s in zip(model.buffers(), buffers):
            b.copy_(s)
    model.zero_grad(set_to_none=True)
    return out


def run(variant, lr, seed, epochs, every, data, device, out, batch=128, probe_size=512):
    path = out / f"{variant}-20-lr{lr:g}-s{seed}.json"
    if path.exists():
        print(f"skip {path.name}")
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr, xte, yte = data
    model = Net(20, variant).to(device)
    fixed = torch.randperm(len(xtr), device=device)[:probe_size]
    px, py = xtr[fixed], ytr[fixed]

    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=0.9, nesterov=True,
    )
    steps_per_epoch = len(xtr) // batch
    total = epochs * steps_per_epoch
    warmup = min(steps_per_epoch, total // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))

    probes = []
    step, start = 0, time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        for k in range(steps_per_epoch):
            if step % every == 0:
                probes.append({"step": step, **probe(model, px, py, ALPHAS)})
            idx = perm[k * batch:(k + 1) * batch]
            loss = F.cross_entropy(model(augment(xtr[idx])), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        last = probes[-1]
        print(f"{variant:>8} lr {lr:g} epoch {epoch + 1}/{epochs} loss {last['loss']:.3f} "
              f"grad change {last['grad_change']} ({time.time() - start:.0f}s)", flush=True)

    path.write_text(json.dumps({
        "variant": variant, "depth": 20, "lr": lr, "seed": seed, "epochs": epochs,
        "every": every, "alphas": ALPHAS, "probe_size": probe_size,
        "steps_per_epoch": steps_per_epoch, "seconds": time.time() - start,
        "probes": probes,
    }, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="+", default=["plain", "plain-bn"])
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "smoothing")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_cifar(device)
    for variant in args.variants:
        run(variant, args.lr, args.seed, args.epochs, args.every, data, device, args.out)


if __name__ == "__main__":
    main()
