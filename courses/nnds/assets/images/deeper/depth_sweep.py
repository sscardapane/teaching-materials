"""Depth sweep for the NNDS lecture on building deeper models.

Trains CIFAR-10 convnets of increasing depth with and without batch
normalization and residual connections, and records what the slides need:

* training loss and test accuracy per epoch (the degradation problem);
* the gradient norm reaching each block at initialization;
* a lesion test: test accuracy after removing one block at a time from the
  trained 56-layer networks (Veit et al., 2016).

The architecture is the CIFAR ResNet of He et al. (2016, Sec. 4.2): a 3x3 stem,
three stages of n blocks with 16, 32 and 64 channels, global average pooling
and a linear classifier, for a depth of 6n + 2. Each block has two 3x3
convolutions. Variants:

    plain       no normalization, no shortcut
    plain-bn    batch normalization, no shortcut
    res         shortcut, no normalization
    res-zero    shortcut, no normalization, last convolution of each branch
                initialized to zero (each block starts as the identity)
    res-bn      shortcut and batch normalization (the standard ResNet)

Convolutions use He initialization by default; --init default keeps PyTorch's
own initialization instead (uniform, about 0.41 times the He scale), so the
opening figure can compare the two. --lr accepts several values; when more
than one initialization or learning rate is swept, file names include them.

Run from the repository root with the repository venv, ideally on a machine
with an Apple GPU or CUDA:

    .venv/bin/python courses/nnds/assets/images/deeper/depth_sweep.py
    .venv/bin/python courses/nnds/assets/images/deeper/depth_sweep.py --quick

Each run is saved as results/<variant>-<depth>-s<seed>.json, and finished runs
are skipped, so the sweep can be interrupted and resumed. plot_depth_sweep.py
turns the results into a review page. CIFAR-10 is downloaded from the
Hugging Face mirror to ~/.cache/cifar10/hf (outside the repository). Needs
torch, numpy, pyarrow and pillow.
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
VARIANTS = ["plain", "plain-bn", "res", "res-zero", "res-bn"]


# ---------------------------------------------------------------------------
# Data: the whole dataset lives on the device, augmentation runs there too.


HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def load_cifar(device, root=Path.home() / ".cache" / "cifar10" / "hf"):
    """CIFAR-10 from the Hugging Face mirror (the Toronto server is very slow)."""
    import io
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    def split(name):
        path = root / f"{name}.parquet"
        if not path.exists():
            root.mkdir(parents=True, exist_ok=True)
            print(f"downloading {name} split to {path}")
            urllib.request.urlretrieve(HF_URL.format(name), path)
        table = pq.read_table(path).to_pydict()
        x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                      for im in table["img"]])
        x = torch.tensor(x).permute(0, 3, 1, 2).float() / 255.0
        return x, torch.tensor(table["label"])

    xtr, ytr = split("train")
    xte, yte = split("test")
    assert xtr.shape == (50000, 3, 32, 32) and xte.shape == (10000, 3, 32, 32)
    mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = xtr.std(dim=(0, 2, 3), keepdim=True)
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    return [t.to(device) for t in (xtr, ytr, xte, yte)]


def augment(x):
    """Random 4-pixel shifts with zero padding and horizontal flips."""
    b = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (b,), device=x.device)
    j = torch.randint(0, 9, (b,), device=x.device)
    rows = (i[:, None] + torch.arange(32, device=x.device))[:, None, :, None]
    cols = (j[:, None] + torch.arange(32, device=x.device))[:, None, None, :]
    bidx = torch.arange(b, device=x.device)[:, None, None, None]
    cidx = torch.arange(3, device=x.device)[None, :, None, None]
    out = xp[bidx, cidx, rows, cols]
    flip = torch.rand(b, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


# ---------------------------------------------------------------------------
# Model


class Block(nn.Module):
    def __init__(self, cin, cout, stride, bn, residual, zero_init, init="he"):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not bn)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not bn)
        self.bn1 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.residual = residual
        self.identity_shortcut = stride == 1 and cin == cout
        self.removable = self.identity_shortcut  # used by the lesion test
        self.removed = False
        if residual and not self.identity_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=not bn),
                nn.BatchNorm2d(cout) if bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()
        if init == "he":
            for conv in (self.conv1, self.conv2):
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        if zero_init:
            nn.init.zeros_(self.conv2.weight)

    def branch(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(h))

    def forward(self, x):
        if self.removed:
            return x
        if self.residual:
            return F.relu(self.branch(x) + self.shortcut(x))
        return F.relu(self.branch(x))


class Net(nn.Module):
    def __init__(self, depth, variant, classes=10, init="he"):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n + 2"
        n = (depth - 2) // 6
        bn = variant in ("plain-bn", "res-bn")
        residual = variant.startswith("res")
        zero_init = variant == "res-zero"
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=not bn)
        if init == "he":
            nn.init.kaiming_normal_(self.stem.weight, nonlinearity="relu")
        self.stem_bn = nn.BatchNorm2d(16) if bn else nn.Identity()
        blocks, cin = [], 16
        for stage, cout in enumerate((16, 32, 64)):
            for k in range(n):
                stride = 2 if (stage > 0 and k == 0) else 1
                blocks.append(Block(cin, cout, stride, bn, residual, zero_init, init))
                cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        return self.head(h.mean(dim=(2, 3)))


# ---------------------------------------------------------------------------
# Measurements


@torch.no_grad()
def evaluate(model, x, y, batch=1000):
    model.eval()
    correct, loss = 0, 0.0
    for k in range(0, len(x), batch):
        logits = model(x[k:k + batch])
        loss += F.cross_entropy(logits, y[k:k + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[k:k + batch]).sum().item()
    model.train()
    return loss / len(x), correct / len(x)


def gradient_norms_at_init(model, x, y):
    """Norm of the loss gradient w.r.t. the first convolution of each block,
    divided by the square root of its number of entries (an RMS)."""
    model.train()
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    norms = []
    for block in model.blocks:
        g = block.conv1.weight.grad
        norms.append((g.norm() / math.sqrt(g.numel())).item())
    model.zero_grad()
    return norms


def lesion(model, x, y):
    accs = []
    for k, block in enumerate(model.blocks):
        if not block.removable:
            accs.append(None)
            continue
        block.removed = True
        accs.append(evaluate(model, x, y)[1])
        block.removed = False
    return accs


# ---------------------------------------------------------------------------
# Training


def run(variant, depth, seed, epochs, data, device, out, lr, batch, lesion_depths,
        init="he", tag=""):
    path = out / f"{variant}-{depth}{tag}-s{seed}.json"
    if path.exists():
        print(f"skip {path.name}")
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr, xte, yte = data
    model = Net(depth, variant, init=init).to(device)
    params = sum(p.numel() for p in model.parameters())

    probe = torch.randperm(len(xtr), device=device)[:256]
    grad_init = gradient_norms_at_init(model, xtr[probe], ytr[probe])

    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=0.9, nesterov=True,
    )
    steps_per_epoch = len(xtr) // batch
    total = epochs * steps_per_epoch
    warmup = min(steps_per_epoch, total // 2)  # linear warmup (one epoch), then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))

    history = {"epoch": [], "train_loss": [], "train_loss_steps": [], "test_loss": [], "test_acc": []}
    diverged = False
    start = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        running = []
        for k in range(steps_per_epoch):
            idx = perm[k * batch:(k + 1) * batch]
            loss = F.cross_entropy(model(augment(xtr[idx])), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running.append(loss.detach())
        running = torch.stack(running).cpu().numpy()
        if not np.isfinite(running).all():
            diverged = True
            print(f"{variant}-{depth}: diverged at epoch {epoch + 1}")
            break
        test_loss, test_acc = evaluate(model, xte, yte)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(running.mean()))
        # 10 points per epoch, for smoother curves in the figure
        history["train_loss_steps"].extend(
            float(c.mean()) for c in np.array_split(running, 10))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"{variant:>8}-{depth:<3} epoch {epoch + 1:>3}/{epochs} "
              f"train {running.mean():.3f} test acc {test_acc:.3f} "
              f"({time.time() - start:.0f}s)", flush=True)

    result = {
        "variant": variant, "depth": depth, "seed": seed, "epochs": epochs,
        "lr": lr, "init": init, "batch": batch, "params": params, "diverged": diverged,
        "seconds": time.time() - start, "grad_norm_init": grad_init,
        "history": history,
    }
    if not diverged and depth in lesion_depths:
        result["baseline_acc"] = evaluate(model, xte, yte)[1]
        result["lesion_acc"] = lesion(model, xte, yte)
    path.write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--depths", nargs="+", type=int, default=[8, 20, 56])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", nargs="+", type=float, default=[0.05])
    ap.add_argument("--init", nargs="+", default=["he"], choices=["he", "default"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lesion-depths", nargs="+", type=int, default=[56])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--quick", action="store_true",
                    help="1 epoch at depths 8 and 20, written to a temporary directory")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.depths = 1, [8, 20]
        args.out = Path(tempfile.gettempdir()) / "depth_sweep_quick"
        args.lesion_depths = [20]
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device {device}, writing to {args.out}")
    data = load_cifar(device)
    for seed in args.seeds:
        for depth in args.depths:
            for variant in args.variants:
                for init in args.init:
                    for lr in args.lr:
                        tag = f"-{init}-lr{lr:g}" if len(args.init) * len(args.lr) > 1 else ""
                        run(variant, depth, seed, args.epochs, data, device, args.out,
                            lr, args.batch, args.lesion_depths, init, tag)


if __name__ == "__main__":
    main()
