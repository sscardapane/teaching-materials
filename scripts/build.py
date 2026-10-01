#!/usr/bin/env python3
"""Build the pilot decks using their original TeX engines and output names."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DECKS = {
    'nn': ('NN2627_Linear_models', '-pdf'),
    'nnds': ('Lecture_3_supervised_learning', '-lualatex'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('course', nargs='?', choices=['all', *DECKS], default='all')
    args = parser.parse_args()
    for course in DECKS if args.course == 'all' else [args.course]:
        name, engine = DECKS[course]
        output = ROOT / 'build' / course
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.pop('PROJECT_ROOT', None)
        command = ['latexmk', '-norc', engine, '-interaction=nonstopmode',
                   '-halt-on-error', '-bibtex-', '-recorder', f'-outdir={output}', f'{name}.tex']
        log = output / 'build.log'
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT / 'courses' / course,
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-6000:])
            raise SystemExit(result.returncode)
        print(output / f'{name}.pdf')


if __name__ == '__main__':
    main()
