"""Mean attention distance per head and layer of two pretrained ViT-B/16.

Reproduces the idea of Figure 7 in Dosovitskiy et al. (2021) for the slide
"What does a ViT attend to?". For each layer and head, we take the attention
weights among the 14x14 patch tokens (the class token is dropped as query and
as key, and each row is renormalized), multiply them by the pixel distance
between patch centers, and average over queries and images.

Models (torchvision vit_b_16, same architecture, 224x224 inputs):
- IMAGENET1K_V1: trained from scratch on ImageNet-1k (1.3M images);
- IMAGENET1K_SWAG_LINEAR_V1: backbone pre-trained on 3.6B Instagram images
  (SWAG, Singh et al. 2022) and frozen; only the classifier saw ImageNet-1k.
Uniform attention over the grid gives about 117 px; a 3x3 neighborhood about 20.

Images: 260 validation images of Imagenette (a 10-class ImageNet subset), 26
per class in file order, from the fast.ai mirror cached in ~/.cache/imagenette.
Each model writes a TikZ fragment with one dot per head, drawn by the slide
inside its own axes (x = layer 1..12, y = distance in pixels). The slide uses
the SWAG model; the ImageNet-only file is kept for comparison (its layer 1 is
already global, and local heads appear in layers 3-5).

    .venv/bin/python courses/nnds/assets/images/transformers/attention_distance.py
"""
import tarfile
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
CACHE = Path.home() / ".cache" / "imagenette"
HERE = Path(__file__).resolve().parent
PER_CLASS = 26
PATCH, GRID = 16, 14

root = CACHE / "imagenette2-160" / "val"
if not root.exists():
    CACHE.mkdir(parents=True, exist_ok=True)
    tgz = CACHE / "imagenette2-160.tgz"
    urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as f:
        f.extractall(CACHE)
files = [p for d in sorted(root.iterdir()) for p in sorted(d.glob("*.JPEG"))[:PER_CLASS]]

# Pixel distance between the centers of every pair of patches, (196, 196).
ij = torch.stack(torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij"), -1)
centers = ij.reshape(-1, 2).float() * PATCH
dist = torch.cdist(centers, centers)


def measure(weights):
    model = vit_b_16(weights=weights).eval()
    prep = weights.transforms()
    inputs = {}
    for i, layer in enumerate(model.encoder.layers):
        layer.ln_1.register_forward_hook(lambda m, a, out, i=i: inputs.__setitem__(i, out))
    n_layers, n_heads = len(model.encoder.layers), model.encoder.layers[0].num_heads
    total = torch.zeros(n_layers, n_heads)
    with torch.no_grad():
        for start in range(0, len(files), 32):
            x = torch.stack([prep(Image.open(f).convert("RGB")) for f in files[start:start + 32]])
            model(x)
            for i, layer in enumerate(model.encoder.layers):
                sa = layer.self_attention
                h = inputs[i]                                   # (B, 197, D) after ln_1
                B, T, D = h.shape
                qkv = h @ sa.in_proj_weight.T + sa.in_proj_bias
                q, k, _ = qkv.reshape(B, T, 3, n_heads, D // n_heads).permute(2, 0, 3, 1, 4)
                a = (q @ k.transpose(-2, -1) / (D // n_heads) ** 0.5).softmax(-1)
                a = a[..., 1:, 1:]                              # patch tokens only
                a = a / a.sum(-1, keepdim=True)
                total[i] += (a * dist).sum(-1).mean(-1).sum(0)  # (heads,)
    return total / len(files)


for name, out in (("IMAGENET1K_V1", "attention_distance_in1k.tex"),
                  ("IMAGENET1K_SWAG_LINEAR_V1", "attention_distance_swag.tex")):
    mean = measure(ViT_B_16_Weights[name])
    lines = [f"% Written by attention_distance.py: ViT-B/16 {name}, {len(files)} Imagenette images.",
             "% Mean attention distance in pixels, one dot per (layer, head)."]
    print(name)
    for i in range(mean.shape[0]):
        print(i + 1, " ".join(f"{v:5.1f}" for v in sorted(mean[i].tolist())))
        lines += [f"\\fill[vitdot] ({i + 1},{v:.1f}) circle (2pt);" for v in mean[i].tolist()]
    (HERE / out).write_text("\n".join(lines) + "\n")
