"""Figures for the NNDS CNN lecture: hand-set edge filters and data augmentation.

Run from the repository root with the repository venv:

    .venv/bin/python courses/nnds/assets/images/cnn/filters/make_figures.py

Needs numpy, matplotlib and scikit-image. The image is the public-domain NASA
portrait shipped with scikit-image (`skimage.data.astronaut`). Everything is
deterministic, so re-running reproduces the tracked PDFs.
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data, transform

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "pdf.fonttype": 42})


def load_gray(size=256):
    img = color.rgb2gray(data.astronaut())
    return transform.resize(img, (size, size), anti_aliasing=True)


def conv2d(x, k):
    """Same-size 2D cross-correlation, as in the slides. We replicate the border
    instead of padding with zeros, so the image frame does not show up as an edge."""
    p = k.shape[0] // 2
    xp = np.pad(x, p, mode="edge")
    out = np.zeros_like(x)
    for u in range(k.shape[0]):
        for v in range(k.shape[1]):
            out += k[u, v] * xp[u:u + x.shape[0], v:v + x.shape[1]]
    return out


def edge_filters():
    x = load_gray()
    vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    horizontal = vertical.T
    panels = [
        (x, "Input", "gray"),
        (np.abs(conv2d(x, vertical)), "Vertical edges", "Greys"),
        (np.abs(conv2d(x, horizontal)), "Horizontal edges", "Greys"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, (img, title, cmap) in zip(axes, panels):
        # Edge maps: dark strokes on white, clipped at the 99th percentile.
        vmax = img.max() if cmap == "gray" else np.percentile(img, 99)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, pad=6, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#999999")
    fig.tight_layout(w_pad=1.0, pad=0.6)
    fig.savefig(OUT / "edge_filters.pdf", transparent=True, metadata={"CreationDate": None},
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def equivariance():
    x = load_gray()
    # Shift down and right, filling the uncovered border with zeros.
    shifted = np.zeros_like(x)
    shifted[30:, 45:] = x[:-30, :-45]
    vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    panels = [
        (x, "Input $X$", "gray"),
        (np.abs(conv2d(x, vertical)), "$f(X)$", "Greys"),
        (shifted, "$T_\\delta X$", "gray"),
        (np.abs(conv2d(shifted, vertical)), "$f(T_\\delta X)$", "Greys"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 6.2))
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        vmax = img.max() if cmap == "gray" else np.percentile(img, 99)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, pad=6, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#999999")
    fig.tight_layout(pad=0.6, h_pad=1.2)
    fig.savefig(OUT / "equivariance.pdf", transparent=True, metadata={"CreationDate": None},
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def augmentation():
    rng = np.random.default_rng(0)
    img = transform.resize(data.astronaut(), (256, 256), anti_aliasing=True)

    def crop(x):
        i, j = rng.integers(0, 64, size=2)
        return transform.resize(x[i:i + 192, j:j + 192], (256, 256), anti_aliasing=True)

    def jitter(x):
        gray = color.rgb2gray(x)[..., None]
        return np.clip(0.6 * x + 0.4 * gray + 0.08, 0, 1) ** 1.2

    panels = [
        (img, "Original"),
        (img[:, ::-1], "Horizontal flip"),
        (crop(img), "Random crop"),
        (jitter(img), "Color jitter"),
        (transform.rotate(img, 15, mode="reflect"), "Small rotation"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8))
    for ax, (x, title) in zip(axes, panels):
        ax.imshow(x)
        ax.set_title(title, pad=6, fontsize=17)
        ax.axis("off")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "augmentation.pdf", transparent=True, metadata={"CreationDate": None},
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    edge_filters()
    equivariance()
    augmentation()
