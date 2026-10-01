#!/usr/bin/env python3
"""Compare every rendered pixel, extracted text, page box and PDF link to fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import PIL
from PIL import Image, ImageChops
import pypdf
from pypdf import PdfReader
from pypdf.generic import IndirectObject
from build import DECKS, ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_structure(path):
    reader = PdfReader(path)
    page_ids = {page.indirect_reference.idnum: i for i, page in enumerate(reader.pages)}

    def normalize(value):
        if isinstance(value, IndirectObject):
            if value.idnum in page_ids:
                return {'page': page_ids[value.idnum]}
            return normalize(value.get_object())
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    pages = []
    for page in reader.pages:
        links = []
        for ref in page.get('/Annots', []):
            annotation = ref.get_object()
            if annotation.get('/Subtype') == '/Link':
                links.append({key: normalize(annotation[key]) for key in
                              ['/Subtype', '/Rect', '/A', '/Dest', '/Border', '/H']
                              if key in annotation})
        pages.append({'media_box': list(page.mediabox), 'crop_box': list(page.cropbox),
                      'rotation': page.rotation, 'links': links})
    destinations = {name: normalize(dest.dest_array)
                    for name, dest in reader.named_destinations.items()}
    metadata = {key: str((reader.metadata or {}).get(key, ''))
                for key in ['/Title', '/Author', '/Subject', '/Keywords']}
    return {'pages': pages, 'destinations': destinations, 'metadata': metadata}


def compare(course, name, dpi, workspace):
    reference = ROOT / 'verification' / 'reference' / f'{name}.pdf'
    candidate = ROOT / 'build' / course / f'{name}.pdf'
    before, after = pdf_structure(reference), pdf_structure(candidate)
    text = [subprocess.check_output(['pdftotext', '-layout', str(path), '-'])
            for path in (reference, candidate)]
    images = []
    for label, path in [('reference', reference), ('candidate', candidate)]:
        prefix = workspace / f'{name}-{label}'
        subprocess.run(['pdftoppm', '-r', str(dpi), '-png', str(path), str(prefix)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        images.append(sorted(workspace.glob(prefix.name + '-*.png'),
                             key=lambda p: int(p.stem.rsplit('-', 1)[1])))
    results = []
    for page, (left, right) in enumerate(zip(*images), 1):
        with Image.open(left) as a, Image.open(right) as b:
            a, b = a.convert('RGB'), b.convert('RGB')
            count = None
            if a.size == b.size:
                bands = ImageChops.difference(a, b).split()
                maximum = ImageChops.lighter(ImageChops.lighter(bands[0], bands[1]), bands[2])
                count = a.width * a.height - maximum.histogram()[0]
            results.append({'page': page, 'reference_size': list(a.size),
                            'candidate_size': list(b.size), 'different_pixels': count})
    equal_count = len(images[0]) == len(images[1]) == len(before['pages']) == len(after['pages'])
    return {'reference_sha256': sha256(reference), 'candidate_sha256': sha256(candidate),
            'pages': len(after['pages']), 'same_page_count': equal_count,
            'text_identical': text[0] == text[1], 'structure_identical': before == after,
            'link_count': sum(len(p['links']) for p in after['pages']),
            'all_pixels_identical': equal_count and all(p['different_pixels'] == 0 for p in results),
            'page_results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--report', type=Path, default=ROOT / 'build' / 'verification.json')
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    report = {'dpi': args.dpi, 'Pillow': PIL.__version__, 'pypdf': pypdf.__version__,
              'tools': {}, 'courses': {}}
    for tool, flag in [('pdftoppm', '-v'), ('pdflatex', '--version'), ('lualatex', '--version')]:
        result = subprocess.run([tool, flag], capture_output=True, text=True, check=True)
        report['tools'][tool] = (result.stdout + result.stderr).splitlines()[0]
    with tempfile.TemporaryDirectory(prefix='teaching-pdf-compare-') as folder:
        for course, names in DECKS.items():
            for name in names:
                key = f'{course}/{name}'
                report['courses'][key] = compare(course, name, args.dpi, Path(folder))
                print(key, {k: v for k, v in report['courses'][key].items() if k != 'page_results'}, flush=True)
    report['passed'] = all(all(result[key] for key in
                              ['same_page_count', 'text_identical', 'structure_identical', 'all_pixels_identical'])
                           for result in report['courses'].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(args.report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
