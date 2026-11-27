"""Frequencies and matrix of the sinusoidal positional embeddings.

For the slide "The sinusoidal embedding matrix" of the Transformer deck. With
D = 64 and base b = 10^4, the left panel plots the D/2 frequencies
omega_j = b^(-2j/D) in log scale, and the right panel the matrix E for the
first 100 positions, with E[t, 2j] = sin(omega_j t), E[t, 2j+1] = cos(omega_j t).

    .venv/bin/python shared/assets/images/transformers/sinusoidal_embeddings.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

D, BASE, T = 64, 1e4, 100
GRAY, BLUE, ORANGE = '#666666', '#6C8EBF', '#D79B00'

j = np.arange(D // 2)
omega = BASE ** (-2 * j / D)
t = np.arange(T)[:, None]
E = np.empty((T, D))
E[:, 0::2] = np.sin(omega * t)
E[:, 1::2] = np.cos(omega * t)

plt.rcParams.update({'font.size': 7, 'axes.edgecolor': GRAY, 'axes.labelcolor': 'black',
                     'xtick.color': GRAY, 'ytick.color': GRAY, 'pdf.fonttype': 42})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.0, 1.35),
                               gridspec_kw={'width_ratios': [1, 1.25], 'wspace': 0.45})

ax1.semilogy(j, omega, 'o-', color=BLUE, markersize=2.5, linewidth=1)
ax1.set_xlabel('hand $j$')
ax1.set_ylabel(r'$\omega_j$')
ax1.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
ax1.grid(True, color='#dddddd', linewidth=0.5)
for side in ('top', 'right'):
    ax1.spines[side].set_visible(False)

cmap = LinearSegmentedColormap.from_list('pe', ['#2f5d8a', BLUE, 'white', ORANGE, '#8a6300'])
im = ax2.imshow(E, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
ax2.set_xlabel('column of $E$')
ax2.set_ylabel('position $t$')
ax2.set_xticks([0, 16, 32, 48, 63])
cbar = fig.colorbar(im, ax=ax2, ticks=[-1, 0, 1], fraction=0.05, pad=0.03)
cbar.outline.set_edgecolor(GRAY)
cbar.ax.tick_params(color=GRAY, labelcolor=GRAY)

fig.savefig(Path(__file__).with_suffix('.pdf'), bbox_inches='tight', transparent=True)
