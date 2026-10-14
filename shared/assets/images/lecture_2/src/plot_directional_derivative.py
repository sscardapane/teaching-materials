"""Gradient and a direction u at a point of a 2D quadratic.

Writes ../directional_derivative.pdf. Written by Claude (2026-10-14).

    .venv/bin/python shared/assets/images/lecture_2/src/plot_directional_derivative.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent
RED, BLUE, ORANGE, GRAY = "#B85450", "#6C8EBF", "#D79B00", "#999999"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 18, "pdf.fonttype": 42})

a, b = np.meshgrid(np.linspace(-2.2, 2.9, 300), np.linspace(-1.6, 2.1, 300))
f = 0.5 * (a ** 2 + 3 * b ** 2)
x = np.array([1.0, 0.55])
g = np.array([x[0], 3 * x[1]])
gn = g / np.linalg.norm(g)
ang = np.deg2rad(115)
rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
u = rot @ gn
tangent = np.array([-gn[1], gn[0]])

fig, ax = plt.subplots(figsize=(6, 4.4))
ax.contour(a, b, f, levels=np.linspace(0.1, 3.5, 9), colors=GRAY, linewidths=0.9)
ax.plot(*np.stack([x - 0.9 * tangent, x + 0.9 * tangent]).T, "--", color=GRAY, linewidth=1.5)
ax.annotate("", xy=x + 0.95 * gn, xytext=x, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.8, mutation_scale=22))
ax.annotate("", xy=x + 0.95 * u, xytext=x, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.8, mutation_scale=22))
ax.text(*(x + 1.05 * gn + np.array([0.02, 0.0])), r"$\nabla f(\mathbf{x})$", color=RED, ha="left", va="center")
ax.text(*(x + 1.05 * u + np.array([-0.05, 0.0])), r"$\mathbf{u}$", color=BLUE, ha="right", va="bottom")
arc = np.linspace(np.arctan2(gn[1], gn[0]), np.arctan2(u[1], u[0]), 30)
ax.plot(x[0] + 0.32 * np.cos(arc), x[1] + 0.32 * np.sin(arc), color=ORANGE, linewidth=2)
mid = arc[len(arc) // 2]
ax.text(x[0] + 0.45 * np.cos(mid), x[1] + 0.45 * np.sin(mid), r"$\alpha$", color=ORANGE, ha="center", va="center")
ax.plot(*x, "o", color="black")
ax.text(*(x + np.array([0.08, -0.22])), r"$\mathbf{x}$")
ax.text(-2.1, -1.5, "dashed: level set,\nwhere $\\mathrm{D}_{\\mathbf{u}}f = 0$", color=GRAY, fontsize=14, va="bottom")
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.savefig(OUT / "directional_derivative.pdf", transparent=True, bbox_inches="tight", pad_inches=0.05,
            metadata={"CreationDate": None})
