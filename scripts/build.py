#!/usr/bin/env python3
"""Build the course decks using their original TeX engines and output names."""
import argparse
import os
import json
import shutil
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DECKS = json.loads((ROOT / 'decks.json').read_text())
ENGINES = {'nn': '-pdf', 'nnds': '-lualatex'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('course', nargs='?', choices=['all', *DECKS], default='all')
    parser.add_argument('--export', action='store_true', help='Copy reviewed builds to tracked pdf/<course>/ downloads')
    args = parser.parse_args()
    for course in DECKS if args.course == 'all' else [args.course]:
        for name in DECKS[course]:
            build(course, name)
            if args.export:
                destination = ROOT / "pdf" / course
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / "build" / course / f"{name}.pdf", destination / f"{name}.pdf")


def build(course, name):
    engine = ENGINES[course]
    output = ROOT / 'build' / course
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('PROJECT_ROOT', None)
    command = ['latexmk', '-norc', engine, '-interaction=nonstopmode',
               '-halt-on-error', '-bibtex-', '-recorder', f'-outdir={output}', f'{name}.tex']
    log = output / f'{name}-build.log'
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT / 'courses' / course,
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-6000:])
        raise SystemExit(result.returncode)
    print(output / f'{name}.pdf')


if __name__ == '__main__':
    main()
