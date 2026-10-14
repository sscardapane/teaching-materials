"""A two-dimensional saddle point: a minimum along one axis, a maximum along the other.

Writes ../saddle_2d.pdf. Written by Claude (2026-10-14).

    .venv/bin/python shared/assets/images/lecture_2/src/plot_saddle_2d.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent
RED, BLUE, GREEN = "#B85450", "#6C8EBF", "#82B366"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "pdf.fonttype": 42})

a, b = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.5, 1.5, 60))
loss = a ** 2 - b ** 2

fig = plt.figure(figsize=(7.2, 3.3))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(a, b, loss, cmap="Blues", alpha=0.55, linewidth=0, antialiased=True, zorder=1)
t = np.linspace(-1.5, 1.5, 50)
ax.plot(t, 0 * t, t ** 2, color=GREEN, linewidth=4, zorder=10, label=r"along $\theta_1$: minimum")
ax.plot(0 * t, t, -t ** 2, color=RED, linewidth=4, zorder=10, label=r"along $\theta_2$: maximum")
ax.scatter([0], [0], [0], color="black", s=40, depthshade=False)
ax.set_xlabel(r"$\theta_1$", labelpad=-8)
ax.set_ylabel(r"$\theta_2$", labelpad=-8)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=28, azim=-55)
ax.set_box_aspect(None, zoom=1.35)

ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(a, b, loss, levels=14, colors="#999999", linewidths=0.8)
s = np.linspace(-1.4, 1.4, 9)
qa, qb = np.meshgrid(s, s)
ax2.quiver(qa, qb, -2 * qa, 2 * qb, color=BLUE, angles="xy", width=0.006, scale=28)
ax2.plot([-1.5, 1.5], [0, 0], color=GREEN, linewidth=3)
ax2.plot([0, 0], [-1.5, 1.5], color=RED, linewidth=3)
ax2.plot(0, 0, "o", color="black")
ax2.set_title(r"$-\nabla\mathcal{L}$ flows in, then out", fontsize=15)
ax2.set_xlabel(r"$\theta_1$"); ax2.set_ylabel(r"$\theta_2$")
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_aspect("equal")

fig.subplots_adjust(left=0.0, right=0.98, bottom=0.06, top=0.92, wspace=-0.05)
fig.savefig(OUT / "saddle_2d.pdf", transparent=True, bbox_inches="tight", pad_inches=0.05,
            metadata={"CreationDate": None})
