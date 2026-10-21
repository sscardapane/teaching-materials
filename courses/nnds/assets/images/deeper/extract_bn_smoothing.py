"""Extract Figure 4(a,b), without redrawing, from Santurkar et al. (2018).

Run with a Python environment containing pypdf. The original PDF stays in the
ignored build directory; the cropped, editable vector PDF is the slide asset.
Source: https://arxiv.org/pdf/1805.11604v5, PDF page 5.
Written by Codex (2026-10-21).
"""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve

from pypdf import PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parents[5]
SOURCE_URL = "https://arxiv.org/pdf/1805.11604v5"
SOURCE_SHA256 = "41d5195d959a8c37f3f23df34023032b2a0361887495cdb558a2402f60437d1e"


def main():
    build = ROOT / "build" / "normalization-review-20261021"
    build.mkdir(parents=True, exist_ok=True)
    source = build / "santurkar-2018.pdf"
    if not source.exists():
        urlretrieve(SOURCE_URL, source)
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source PDF differs from the reviewed arXiv v5 file")
    page = PdfReader(source).pages[4]
    # PDF points, bottom-left origin. Both panels retain their axes, legends
    # and original (a,b) labels. Excludes panel (c) and the paper's caption.
    left, bottom, right, top = 105, 610, 367, 720
    page.add_transformation(Transformation().translate(-left, -bottom))
    writer = PdfWriter()
    output = writer.add_blank_page(width=right - left, height=top - bottom)
    output.merge_page(page)
    destination = Path(__file__).with_name("bn_smoothing.pdf")
    with destination.open("wb") as stream:
        writer.write(stream)
    print(f"Source SHA256: {sha256(source.read_bytes()).hexdigest()}")
    print(destination)


if __name__ == "__main__":
    main()
