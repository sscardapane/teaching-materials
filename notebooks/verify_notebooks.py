"""Execute the shared notebooks (PT01 to PT05) and check the student, instructor and solution paths.

For every lab this script checks that:

- the student and solutions notebooks match their instructor source (make_student_notebook.py);
- the unfilled student notebook stops at the first activity (PT01 runs to the end);
- the instructor notebook runs without errors;
- the student notebook, completed with the solutions, runs without errors, and its
  checks reject plausible wrong implementations (verify_regressions.py).

Usage: python verify_notebooks.py [--write-outputs] [--only PT03 PT04 ...]
--write-outputs saves the instructor outputs and refreshes the HTML previews of
the student and solutions notebooks. Requires nbformat, nbclient, nbconvert,
ipykernel and the notebook dependencies.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from make_student_notebook import (INSTRUCTOR, LABS, RELEASED, build_solutions, build_student,
                                   clear_execution, match_exercises, write)
from verify_regressions import INSTRUCTOR_CHECKS, REGRESSIONS, extra_checks

ROOT = Path(__file__).resolve().parent
# Text of the error that the unfilled student notebook must stop at (None: runs to the end).
FIRST_STOP = {
    "PT01_Introduction_to_PyTorch": None,
    "PT02_Logistic_regression": "Activity 1: write linear_logits",
    "PT03_Automatic_differentiation": "Complete Activity 1a",
    "PT04_Convolutional_neural_networks": "Complete Activity 1",
    "PT05_Training_deeper_networks": "Complete Activity 1",
}
# The completed student path checks the code, not the training curves: shorter runs.
QUICK_RUNS = {
    "PT04_Convolutional_neural_networks": {"EPOCHS = 4": "EPOCHS = 1"},
    "PT05_Training_deeper_networks": {"DEPTHS, SUBSET = [2, 8, 16], 10_000": "DEPTHS, SUBSET = [2], 2_000",
                                      "REGULARIZATION_EPOCHS = 6": "REGULARIZATION_EPOCHS = 1"},
}


def execute(nb):
    return NotebookClient(nb, timeout=900, kernel_name="nnds-validation",
                          resources={"metadata": {"path": str(ROOT)}}).execute()


def as_node(notebook):
    return nbformat.reads(json.dumps(notebook), as_version=4)


def write_html(notebook, path):
    html, _ = HTMLExporter().from_notebook_node(notebook)
    path.write_text(html)


def verify_lab(stem, write_outputs):
    number = stem.split("_")[0]
    source_path = INSTRUCTOR / f"{stem}.ipynb"
    instructor = nbformat.read(source_path, as_version=4)
    nbformat.validate(instructor)

    student = build_student(stem)
    committed = clear_execution(json.loads((ROOT / f"{stem}.ipynb").read_text()))
    assert committed == student, f"Rebuild {stem}.ipynb with make_student_notebook.py"
    student = as_node(student)
    nbformat.validate(student)
    assert not any("reference" in c.metadata.get("tags", []) for c in student.cells)
    if stem in RELEASED:
        solutions = build_solutions(stem)
        assert clear_execution(json.loads((ROOT / f"{stem}_solutions.ipynb").read_text())) == solutions, \
            f"Rebuild {stem}_solutions.ipynb with make_student_notebook.py"
        nbformat.validate(as_node(solutions))
    print(f"PASS {number}: generated student and solutions notebooks", flush=True)

    stop = FIRST_STOP[stem]
    try:
        execute(copy.deepcopy(student))
    except CellExecutionError as error:
        assert stop and "NotImplementedError" in str(error) and stop in str(error), str(error)[-2000:]
    else:
        assert stop is None, f"The unfilled {number} student notebook did not stop at '{stop}'"
    print(f"PASS {number}: unfilled student notebook " + (f"stops at '{stop}'" if stop else "runs to the end"), flush=True)

    checked = copy.deepcopy(instructor)
    if stem in INSTRUCTOR_CHECKS:
        checked.cells.append(nbformat.v4.new_code_cell(INSTRUCTOR_CHECKS[stem], id="instructor-verification"))
    executed = execute(checked)
    if stem in INSTRUCTOR_CHECKS:
        executed.cells.pop()
    print(f"PASS {number}: instructor notebook runs, including the optional sections", flush=True)

    matches = match_exercises(json.loads(source_path.read_text()))
    solution_sources = {cell.id: cell.source for cell in instructor.cells}
    completed = copy.deepcopy(student)
    for cell in completed.cells:
        if cell.id in matches:
            cell.source = solution_sources[matches[cell.id]]
        for full, quick in QUICK_RUNS.get(stem, {}).items():
            cell.source = cell.source.replace(full, quick)
    sources = "\n".join(cell.source for cell in completed.cells)
    assert all(quick in sources for quick in QUICK_RUNS.get(stem, {}).values()), "A training setting was not shortened"
    completed.cells.append(nbformat.v4.new_code_cell(REGRESSIONS[stem], id="private-verification"))
    execute(completed)
    print(f"PASS {number}: completed student notebook runs and its checks reject wrong implementations", flush=True)

    for name, notebook in extra_checks(stem, executed):
        execute(notebook)
        print(f"PASS {number}: {name}", flush=True)

    if write_outputs:
        executed.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(executed, source_path)
        # Execution can change the notebook metadata: regenerate from the saved source.
        generated = [(ROOT / f"{stem}.ipynb", build_student(stem))]
        if stem in RELEASED:
            generated.append((ROOT / f"{stem}_solutions.ipynb", build_solutions(stem)))
        for path, notebook in generated:
            write(path, notebook)
            write_html(as_node(notebook), path.with_suffix(".html"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="PTxx", help="Verify only these labs, e.g. --only PT03 PT04")
    args = parser.parse_args()
    labs = [stem for stem in LABS if not args.only or stem.split("_")[0] in args.only]
    with tempfile.TemporaryDirectory(prefix="nnds-validation-") as temp:
        root = Path(temp)
        kernel = root / "kernels" / "nnds-validation"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "NNDS validation", "language": "python"}))
        os.environ.update(JUPYTER_PATH=str(root), JUPYTER_RUNTIME_DIR=str(root / "runtime"),
                          IPYTHONDIR=str(root / "ipython"), MPLCONFIGDIR=str(root / "mpl"),
                          OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for stem in labs:
            verify_lab(stem, args.write_outputs)


if __name__ == "__main__":
    main()
