"""Gradient descent, momentum and Adam on a poorly conditioned quadratic.

Writes ../conditioning.pdf (gradient descent alone), ../momentum_path.pdf
(gradient descent and momentum) and ../adam_path.pdf (all three, same start). Written by Claude
(2026-10-14) for the NNDS derivatives and optimization deck.

    .venv/bin/python shared/assets/images/lecture_2/src/plot_conditioning.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent
RED, BLUE, GREEN, GRAY = "#B85450", "#6C8EBF", "#82B366", "#999999"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 19, "pdf.fonttype": 42})

KAPPA = 20.0
START = np.array([-4.5, 1.0])


def grad(theta):
    return np.array([theta[0], KAPPA * theta[1]])


# Each method runs until it is within TOL of the minimum, so every path ends at the star.
TOL = 0.05
MAX_STEPS = 500


def run(eta, lam=0.0):
    theta, u, path = START.copy(), np.zeros(2), [START.copy()]
    while np.linalg.norm(theta) > TOL and len(path) <= MAX_STEPS:
        u = lam * u - eta * grad(theta)
        theta = theta + u
        path.append(theta.copy())
    return np.array(path)


def adam(eta, b1=0.9, b2=0.999, eps=1e-8):
    theta, m, v, path = START.copy(), np.zeros(2), np.zeros(2), [START.copy()]
    t = 0
    while np.linalg.norm(theta) > TOL and len(path) <= MAX_STEPS:
        t += 1
        g = grad(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - eta * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        path.append(theta.copy())
    return np.array(path)


def base(ax):
    a, b = np.meshgrid(np.linspace(-5.2, 1.2, 300), np.linspace(-1.3, 1.3, 300))
    loss = 0.5 * (a ** 2 + KAPPA * b ** 2)
    ax.contour(a, b, loss, levels=np.geomspace(0.05, 40, 9), colors=GRAY, linewidths=0.8)
    ax.plot(0, 0, marker="*", color="black", markersize=13, zorder=5)
    ax.set_xlabel(r"$\theta_1$ (flat direction)")
    ax.set_ylabel(r"$\theta_2$ (steep)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def draw(ax, path, color, label):
    ax.plot(path[:, 0], path[:, 1], "-o", color=color, markersize=3, linewidth=1.5, label=label)
    print(f"{label}: {len(path) - 1} steps")


def save(fig, name):
    fig.savefig(OUT / name, transparent=True, bbox_inches="tight", pad_inches=0.05,
                metadata={"CreationDate": None})
    plt.close(fig)


# Step size close to the stability limit 2 / KAPPA of the steep direction.
gd = run(eta=0.095)

fig, ax = plt.subplots(figsize=(7, 3.4))
base(ax)
draw(ax, gd, RED, "Gradient descent")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save(fig, "conditioning.pdf")

fig, ax = plt.subplots(figsize=(7, 3.4))
base(ax)
draw(ax, gd, RED, "Gradient descent")
draw(ax, run(eta=0.04, lam=0.7), BLUE, "Momentum")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save(fig, "momentum_path.pdf")

fig, ax = plt.subplots(figsize=(7, 3.4))
base(ax)
draw(ax, gd, RED, "GD")
draw(ax, run(eta=0.04, lam=0.7), BLUE, "Momentum")
draw(ax, adam(eta=0.15), GREEN, "Adam")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
save(fig, "adam_path.pdf")
