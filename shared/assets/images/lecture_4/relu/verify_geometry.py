"""Check the mathematics of the two NN ReLU figures; Python standard library only.

Run from any directory: python3 path/to/verify_geometry.py
The figure polygons are read from regions.tex, so this also checks their geometry.
"""
from fractions import Fraction as F
from pathlib import Path
import re


def relu(x):
    return max(0, x)


def triangle(x):
    return 2 * relu(x) - 4 * relu(x - F(1, 2)) + 2 * relu(x - 1)


def compose(x, k):
    for _ in range(k):
        x = triangle(x)
    return x


def inside_convex(point, vertices):
    cross = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        cross.append((b[0]-a[0])*(point[1]-a[1]) - (b[1]-a[1])*(point[0]-a[0]))
    return all(c >= 0 for c in cross) or all(c <= 0 for c in cross)


source = Path(__file__).with_name('regions.tex').read_text()
paths = re.findall(r'\\fill\[[^\]]+\] (.*?);', source)
assert len(paths) == 4
polygons = [[tuple(map(F, pair)) for pair in re.findall(r'\((-?\d+),(-?\d+)\)', path)] for path in paths]
gates = [(0, 0), (0, 1), (1, 0), (1, 1)]
checked = 0
for i in range(-40, 41):
    for j in range(-40, 41):
        x, y = F(i, 20), F(j, 20)
        expected = relu(x) + relu(y-x)
        formulas = [F(0), y-x, x, y]
        members = [index for index, poly in enumerate(polygons) if inside_convex((x, y), poly)]
        assert members, (x, y)
        for index in members:
            assert formulas[index] == expected, (x, y, index)
            if x != 0 and y != x:
                assert gates[index] == (int(x > 0), int(y > x)), (x, y, index)
        checked += 1
print(f'Activation regions: {checked} rational inputs, including shared boundaries, match the plotted formulas and gates.')

for k in range(1, 9):
    pieces = 2**k
    # Check exact vertices, slopes and interior points against repeated ReLU evaluation.
    for j in range(pieces + 1):
        assert compose(F(j, pieces), k) == j % 2
    for j in range(pieces):
        for t in (F(1, 7), F(1, 3), F(1, 2), F(5, 6)):
            expected = t if j % 2 == 0 else 1-t
            assert compose((j+t)/pieces, k) == expected
    for i in range(1001):
        x = F(i, 1000)
        # Realize the composition with k width-3 hidden layers: every layer
        # after the first receives the preceding layer's affine readout.
        h = [relu(x), relu(x-F(1, 2)), relu(x-1)]
        for _ in range(k-1):
            z = 2*h[0]-4*h[1]+2*h[2]
            h = [relu(z), relu(z-F(1, 2)), relu(z-1)]
        assert 2*h[0]-4*h[1]+2*h[2] == compose(x, k)
    print(f'Composition {k}: {pieces} affine pieces; width-3 network and plotted vertices agree exactly.')
assert triangle(F(-1)) == triangle(F(2)) == 0
print('All geometry checks passed.')
