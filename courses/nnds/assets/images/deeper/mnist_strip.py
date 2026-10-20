"""Strip of MNIST test digits (one per class) for the LeNet slide.

Dark strokes on a transparent background, so the image sits on the slide
canvas. MNIST test split from the Hugging Face mirror (ylecun/mnist), cached
in ~/.cache/mnist. Run with the repository venv (needs pyarrow and pillow).
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

URL = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet"
CACHE = Path.home() / ".cache" / "mnist" / "test.parquet"
OUT = Path(__file__).resolve().parent / "mnist_strip.png"

if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, CACHE)
table = pq.read_table(CACHE).to_pydict()
first = {}
for im, label in zip(table["image"], table["label"]):
    first.setdefault(label, im["bytes"])
    if len(first) == 10:
        break

scale, gap = 4, 8  # 28x28 digits drawn at 112x112, 8 px apart
tiles = []
for d in range(10):
    x = np.asarray(Image.open(io.BytesIO(first[d])).convert("L"), dtype=np.uint8)
    rgba = np.zeros((28, 28, 4), dtype=np.uint8)
    rgba[..., :3] = 0x33           # dark gray strokes
    rgba[..., 3] = x               # ink intensity becomes opacity
    tiles.append(Image.fromarray(rgba, "RGBA").resize((28 * scale,) * 2, Image.NEAREST))
w = 10 * 28 * scale + 9 * gap
strip = Image.new("RGBA", (w, 28 * scale), (0, 0, 0, 0))
for d, tile in enumerate(tiles):
    strip.paste(tile, (d * (28 * scale + gap), 0))
strip.save(OUT)
print(OUT)
