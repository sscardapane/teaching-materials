#!/usr/bin/env python3
"""Build static teaching pages locally; this script does not publish them."""
import argparse
import html
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='/teaching-materials/')
    args = parser.parse_args()
    base = '/' + args.base.strip('/') + '/'
    output = ROOT / 'html' / 'dist'
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for source in sorted((ROOT / 'html' / 'slides').glob('*.md')):
        name = source.stem
        subprocess.run(['npx', '--no-install', 'slidev', 'build', str(source),
                        '--base', f'{base}nnds/{name}/', '--out', str(output / 'nnds' / name)],
                       cwd=ROOT / 'html', check=True)
        items.append((f'nnds/{name}/', name.replace('_', ' ')))
    for course in ['nn', 'nnds']:
        for pdf in sorted((ROOT / 'pdf' / course).glob('*.pdf')):
            target = output / 'pdf' / course / pdf.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, target)
            items.append((f'pdf/{course}/{pdf.name}', f'{course.upper()}: {pdf.stem} (PDF)'))
    notebook_output = output / 'notebooks'
    notebook_output.mkdir(exist_ok=True)
    for source in sorted((ROOT / 'notebooks').iterdir()):
        if source.suffix in {'.ipynb', '.html'}:
            shutil.copy2(source, notebook_output / source.name)
            items.append((f'notebooks/{source.name}', source.name))
            if source.suffix == '.ipynb':
                colab = 'https://colab.research.google.com/github/sscardapane/teaching-materials/blob/main/notebooks/'
                items.append((colab + source.name, source.stem.replace('_', ' ') + ' (Open in Colab)'))
    shutil.copytree(ROOT / 'notebooks' / 'data', notebook_output / 'data', dirs_exist_ok=True)
    rows = '\n'.join(f'<li><a href="{html.escape(url)}">{html.escape(title)}</a></li>' for url, title in items)
    (output / 'index.html').write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Teaching materials</title></head><body><main>'
        '<h1>Teaching materials</h1><p>Neural Networks and Neural Networks for Data Science.</p>'
        '<p><a href="https://github.com/sscardapane/teaching-materials">Sources and instructions</a></p>'
        f'<ul>{rows}</ul></main></body></html>\n')
    print(output)


if __name__ == '__main__':
    main()
