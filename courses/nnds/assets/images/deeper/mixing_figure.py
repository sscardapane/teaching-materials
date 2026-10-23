"""Strip of CIFAR-10 images for the MixUp/CutMix slide.

Four tiles: a cat, a dog, their MixUp with lambda = 0.7 and a CutMix that
pastes a 16x16 patch of the dog (a quarter of the image, so lambda = 0.75 for
the cat). The slide writes the labels under each tile. CIFAR-10 test split from
the Hugging Face mirror, cached in ~/.cache/cifar10/hf (the same cache as
depth_sweep.py). Run with the repository venv (needs pyarrow and pillow).
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
CACHE = Path.home() / ".cache" / "cifar10" / "hf" / "test.parquet"
OUT = Path(__file__).resolve().parent / "mixing.png"
CAT, DOG = 3, 5
LAMBDA = 0.7
PATCH = (8, 24, 8, 24)  # rows and columns of the pasted patch

if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, CACHE)
table = pq.read_table(CACHE).to_pydict()


def first(label):
    for im, y in zip(table["img"], table["label"]):
        if y == label:
            return np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")).astype(float)


x1, x2 = first(CAT), first(DOG)
mixup = LAMBDA * x1 + (1 - LAMBDA) * x2
cutmix = x1.copy()
r0, r1, c0, c1 = PATCH
cutmix[r0:r1, c0:c1] = x2[r0:r1, c0:c1]

scale, gap = 5, 24  # 32x32 images drawn at 160x160
tiles = [x1, x2, mixup, cutmix]
w = len(tiles) * 32 * scale + (len(tiles) - 1) * gap
strip = Image.new("RGBA", (w, 32 * scale), (0, 0, 0, 0))
for k, t in enumerate(tiles):
    tile = Image.fromarray(t.round().astype(np.uint8)).resize((32 * scale,) * 2, Image.NEAREST)
    strip.paste(tile, (k * (32 * scale + gap), 0))
strip.save(OUT)
print(f"wrote {OUT}")
