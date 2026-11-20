"""Build the student and solutions notebooks (PT01 to PT05) from the instructor sources.

The instructor notebooks in `instructor/` are the only files to edit:

- cells tagged `reference` are instructor-only and never reach students;
- cells tagged `reference` and `solution` form the solutions notebook;
- code cells tagged `exercise` are the stubs students complete. Every stub must be
  redefined, with the same signatures, by a solution code cell that follows it.

The student notebook is the instructor notebook without reference cells. The
solutions notebook repeats the lab's headings above each solution, so it has the
same outline as the lab. It is written only for labs listed in RELEASED.

Use --check to verify that the committed notebooks match their instructor sources.
"""

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTRUCTOR = ROOT / "instructor"
LABS = [
    "PT01_Introduction_to_PyTorch",
    "PT02_Logistic_regression",
    "PT03_Automatic_differentiation",
    "PT04_Convolutional_neural_networks",
    "PT05_Training_deeper_networks",
]
# Solutions are published after the lab has taken place. On November 20, 2026:
# NNDS held PT01, PT02, PT04 and PT05 (the two parts of the old PT04), NN held PT03.
RELEASED = set(LABS)
COLAB = "https://colab.research.google.com/github/sscardapane/teaching-materials/blob/main/notebooks/"


def lines(text):
    return text.splitlines(keepends=True)


def tags(cell):
    return cell.get("metadata", {}).get("tags", [])


def source(cell):
    return "".join(cell["source"])


def clear_execution(notebook):
    for cell in notebook["cells"]:
        cell["metadata"].pop("execution", None)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def load_instructor(stem):
    return json.loads((INSTRUCTOR / f"{stem}.ipynb").read_text())


def build_student(stem):
    notebook = load_instructor(stem)
    notebook["cells"] = [cell for cell in notebook["cells"] if "reference" not in tags(cell)]
    return clear_execution(notebook)


def definitions(code):
    """Top-level functions and classes with their argument lists (methods for classes)."""
    found = {}
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = ast.unparse(node.args)
        elif isinstance(node, ast.ClassDef):
            found[node.name] = {item.name: ast.unparse(item.args) for item in node.body
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return found


def match_exercises(notebook):
    """Map every exercise cell to the solution cell that completes it."""
    matches, pending = {}, []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "exercise" in tags(cell):
            stub = definitions(source(cell))
            if not stub:
                raise RuntimeError(f"Exercise cell {cell['id']} defines no function or class")
            pending.append((cell["id"], stub))
        elif "solution" in tags(cell):
            solved = definitions(source(cell))
            for exercise_id, stub in list(pending):
                if not set(stub) <= set(solved):
                    continue
                for name, signature in stub.items():
                    actual = solved[name]
                    if isinstance(signature, dict):
                        if not isinstance(actual, dict) or any(actual.get(m) != a for m, a in signature.items()):
                            raise RuntimeError(f"{cell['id']}: class {name} differs from the stub in {exercise_id}")
                    elif actual != signature:
                        raise RuntimeError(f"{cell['id']}: {name}({actual}) differs from the stub ({signature})")
                matches[exercise_id] = cell["id"]
                pending.remove((exercise_id, stub))
    if pending:
        raise RuntimeError(f"Exercises without a solution: {[exercise for exercise, _ in pending]}")
    return matches


def build_solutions(stem):
    notebook = load_instructor(stem)
    match_exercises(notebook)
    title = source(notebook["cells"][0]).splitlines()[0].lstrip("# ").strip()
    number = stem.split("_")[0]
    cells = [{"cell_type": "markdown", "id": "solutions-intro", "metadata": {}, "source": lines(
        f"# {title}: solutions\n\n"
        f"The solutions to the exercises of [{number}]({COLAB}{stem}.ipynb). Each solution "
        "comes under the heading it has in the lab, with a short explanation followed by "
        "the code. If you fall behind during the lab, copy the solution cell into your "
        "notebook in place of your version and continue from there.")}]
    path, emitted = {}, {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and "reference" not in tags(cell):
            for line in source(cell).splitlines():
                level = len(line) - len(line.lstrip("#"))
                if level in (2, 3) and line[level:level + 1] == " ":
                    path[level] = line[level:].strip()
                    if level == 2:
                        path.pop(3, None)
        if "solution" not in tags(cell):
            continue
        for level in (2, 3):
            if path.get(level) and (path.get(level) != emitted.get(level)
                                    or (level == 3 and path.get(2) != emitted.get(2))):
                cells.append({"cell_type": "markdown", "id": f"heading-{len(cells)}", "metadata": {},
                              "source": lines("#" * level + " " + path[level])})
        emitted = dict(path)
        copied = json.loads(json.dumps(cell))
        copied["metadata"] = {}
        cells.append(copied)
    notebook["cells"] = cells
    return clear_execution(notebook)


def outputs():
    for stem in LABS:
        yield ROOT / f"{stem}.ipynb", build_student(stem)
        if stem in RELEASED:
            yield ROOT / f"{stem}_solutions.ipynb", build_solutions(stem)


def write(path, notebook):
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for stem in LABS:
        if stem not in RELEASED and (ROOT / f"{stem}_solutions.ipynb").exists():
            raise SystemExit(f"{stem} is not released: remove its solutions notebook.")
    for path, notebook in outputs():
        if args.check:
            existing = clear_execution(json.loads(path.read_text())) if path.exists() else None
            if existing != notebook:
                raise SystemExit(f"Rebuild {path.name}: it differs from its instructor source.")
            print(f"PASS generated: {path.name}")
        else:
            write(path, notebook)
            print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
