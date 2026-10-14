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
    parser.add_argument('--file', type=Path, help='Build one registered course wrapper (for editor recipes)')
    parser.add_argument('--outdir', type=Path, help='Override the build output directory')
    parser.add_argument('--outdir-base', type=Path, help='Write each course to <dir>/<course> (editor layout)')
    parser.add_argument('--whole-course', action='store_true', help='With --file, build every deck of that course')
    parser.add_argument('--synctex', action='store_true', help='Generate source/PDF navigation data')
    args = parser.parse_args()
    selected = [(course, name)
                for course in DECKS if args.course in ('all', course)
                for name in DECKS[course]]
    if args.file:
        matches = [(course, name) for course, name in selected
                   if (ROOT / 'courses' / course / f'{name}.tex').resolve() == args.file.resolve()]
        if not matches:
            parser.error('--file must be a registered course wrapper from decks.json')
        if args.whole_course:
            selected = [(course, name) for course, name in selected if course == matches[0][0]]
        else:
            selected = matches
    for course, name in selected:
        outdir = args.outdir_base / course if args.outdir_base else args.outdir
        output = build(course, name, outdir=outdir, synctex=args.synctex)
        if args.export:
            destination = ROOT / 'pdf' / course
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, destination / output.name)


def build(course, name, *, outdir=None, synctex=False):
    engine = ENGINES[course]
    output = outdir.resolve() if outdir else ROOT / 'build' / course
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('PROJECT_ROOT', None)
    command = ['latexmk', '-norc', engine, '-interaction=nonstopmode',
               '-halt-on-error', '-file-line-error', '-bibtex-', '-recorder',
               f'-outdir={output}', f'{name}.tex']
    if synctex:
        command.insert(2, '-synctex=1')
    log = output / f'{name}-build.log'
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT / 'courses' / course,
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-6000:])
        raise SystemExit(result.returncode)
    pdf = output / f'{name}.pdf'
    overfull = (output / f'{name}.log').read_text(errors='replace').count('Overfull \\vbox')
    print(f'{pdf}  ({overfull} overfull vbox warnings)' if overfull else pdf)
    return pdf


if __name__ == '__main__':
    main()
