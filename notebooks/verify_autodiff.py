"""Autodiff execution, student export and regression checks for verify_notebooks.py."""
import copy
import json
from pathlib import Path

import nbformat
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from make_student_notebook import build_autodiff, clear_execution

ROOT = Path(__file__).resolve().parent
NAME = "PT03_Automatic_differentiation"
SOLUTIONS = ROOT / f"{NAME}_solutions.ipynb"
STUDENT = ROOT / f"{NAME}.ipynb"
EXERCISES = {"ad-local-exercise": "ad-local-reference",
             "ad-nonlinear-exercise": "ad-nonlinear-reference",
             "ad-backward-exercise": "ad-backward-reference",
             "ad-step-exercise": "ad-step-reference",
             "ad-mlp-exercise": "ad-mlp-reference",
             "ad-grad-exercise": "ad-grad-reference",
             "ad-jacobian-exercise": "ad-jacobian-reference",
             "ad-topo-exercise": "ad-topo-reference"}


REGRESSION_CHECKS = r'''
# These checks are private verification cells, not distributed answers.
def must_reject(function):
    try:
        function()
    except (AssertionError, ValueError):
        return
    raise AssertionError('The notebook checks accepted an incorrect implementation')

saved_reduce = sum_to_shape
def wrong_axis(v, shape):
    if np.asarray(v).ndim == 2 and len(shape) == 1:
        return np.asarray(v).sum(axis=1)
    return saved_reduce(v, shape)
sum_to_shape = wrong_axis
must_reject(check_broadcasting)
sum_to_shape = saved_reduce

saved_matmul = matmul_vjp
def reversed_outer(a, b, v):
    if a.ndim == 2 and b.ndim == 1:
        return np.outer(b, v), a.T @ v
    return saved_matmul(a, b, v)
matmul_vjp = reversed_outer
must_reject(check_matmul_vjp)
matmul_vjp = saved_matmul

saved_multiply = multiply_vjp
multiply_vjp = lambda a, b, v: (v * a, v * b)
must_reject(check_elementwise_vjps)
multiply_vjp = saved_multiply

saved_tanh, saved_relu = tanh_vjp, relu_vjp
tanh_vjp = lambda y, v: v / np.cosh(y) ** 2  # Applies the input formula to the saved output.
must_reject(check_nonlinear_vjps)
tanh_vjp = saved_tanh
relu_vjp = lambda x, v: (x >= 0) * v  # Passes the adjoint through at exactly zero.
must_reject(check_nonlinear_vjps)
relu_vjp = saved_relu

saved_backward = Tensor.backward
def overwriting_backward(self, gradient=None):
    adjoints = {self: initial_adjoint(self, gradient)}
    for node in reversed(topological_order(self)):
        if node.vjp is None:
            node.grad = adjoints[node].copy() if node.grad is None else node.grad + adjoints[node]
            continue
        for parent, contribution in zip(node.parents, node.vjp(adjoints[node])):
            if parent.requires_grad:
                adjoints[parent] = np.asarray(contribution).copy()
Tensor.backward = overwriting_backward
must_reject(check_backward)
Tensor.backward = lambda self, gradient=None: premature_backward(self)
must_reject(check_backward)
def resetting_backward(self, gradient=None):  # Overwrites .grad instead of accumulating.
    for node in topological_order(self):
        node.grad = None
    saved_backward(self, gradient)
Tensor.backward = resetting_backward
must_reject(check_backward)
def retaining_backward(self, gradient=None):  # Stores .grad on intermediate nodes too.
    saved_backward(self, gradient)
    for node in topological_order(self):
        if node.vjp is not None:
            node.grad = np.zeros_like(node.data)
Tensor.backward = retaining_backward
must_reject(check_backward)
Tensor.backward = saved_backward

must_reject(lambda: check_topological_order(
    lambda output: [node for node in graph_nodes(output) if node.requires_grad]))
must_reject(lambda: check_topological_order(lambda output: topological_order(output)[::-1]))
must_reject(lambda: check_topological_order(
    lambda output: list(reversed(graph_nodes(output)))))  # Includes constants.

saved_step = classifier_step
def frozen_bias(X, targets, W, b, lr):
    result = saved_step(X, targets, W, b, lr)
    result['b'] = b.copy()
    return result
classifier_step = frozen_bias
must_reject(check_classifier_step)
classifier_step = saved_step

saved_ce = cross_entropy
def sum_loss(logits, targets):
    return saved_ce(logits, targets) * len(targets)
cross_entropy = sum_loss
must_reject(check_classifier_step)
cross_entropy = saved_ce

# Supplied primitives: non-unit seeds, offsets, validation and copies.
z = Tensor([[1000., 1001., -999.], [10., 11., 12.]], requires_grad=True)
t = np.array([0, 2])
shift = np.array([[1e12], [-1e12]])
close(cross_entropy(Tensor(z.data + shift), t).data, cross_entropy(z, t).data)
cross_entropy(z, t).backward(np.array(3.))
tripled = z.grad
z.grad = None
cross_entropy(z, t).backward()
close(tripled, 3 * z.grad)
expect_error(ValueError, lambda: cross_entropy(z, np.array([0., 1.])))
expect_error(ValueError, lambda: cross_entropy(z, np.array([0, 9])))
expect_error(ValueError, lambda: cross_entropy(Tensor(np.empty((0, 3))), np.array([], dtype=int)))
expect_error(ValueError, lambda: Tensor(np.ones((2, 2, 2))) @ Tensor([1., 2.]))
expect_error(ValueError, lambda: sum_to_shape(np.ones((2, 3)), (2,)))
close(sum_to_shape(np.arange(1., 10.).reshape(3, 3), (3, 1)), [[6.], [15.], [24.]])
close(sum_to_shape(np.ones((2, 3, 4)), (1, 3, 1)), np.full((1, 3, 1), 8.))
close(sum_to_shape(np.arange(9.).reshape(3, 3), ()), 36.)
value = np.array([1., 2.])
copied = Tensor(value)
value[0] = 100
close(copied.data, np.array([1., 2.]))
assert not (Tensor(1.) + Tensor(2.)).requires_grad
assert (Tensor(1., requires_grad=True) * 2.).requires_grad
assert (3. - Tensor(1., requires_grad=True)).requires_grad

# More broadcast / reduction edge cases, checked independently with PyTorch.
for a_shape, b_shape in [((2, 1, 4), (1, 3, 1)), ((), ()), ((1, 1), (3, 4))]:
    gen = np.random.default_rng(91)
    a = Tensor(gen.normal(size=a_shape), requires_grad=True)
    b = Tensor(gen.normal(size=b_shape), requires_grad=True)
    (a * b + a).sum().backward()
    ta = torch.tensor(a.data, requires_grad=True)
    tb = torch.tensor(b.data, requires_grad=True)
    ((ta * tb) + ta).sum().backward()
    close(a.grad, ta.grad.numpy())
    close(b.grad, tb.grad.numpy())

matrix = Tensor(np.arange(6.).reshape(2, 3), requires_grad=True)
matrix.sum(axis=-1).backward(np.array([2., 3.]))
close(matrix.grad, np.array([[2., 2., 2.], [3., 3., 3.]]))
matrix.grad = None
matrix.sum(axis=()).backward(np.ones((2, 3)))
close(matrix.grad, np.ones((2, 3)))
cube = Tensor(np.arange(24.).reshape(2, 3, 4), requires_grad=True)
cube.sum(axis=(0, -1)).backward(np.array([1., 2., 3.]))
close(cube.grad, np.broadcast_to(np.array([1., 2., 3.])[None, :, None], (2, 3, 4)))
scalar = Tensor(2., requires_grad=True)
scalar.sum().backward()
close(scalar.grad, np.array(1.))
close(grad(lambda x: x * x)(np.array(3.)), np.array(6.))
close(jacobian(lambda x: x * x, np.array(3.)), np.array(6.))
close(jacobian(lambda x: Tensor(np.array([])), np.array([1., 2.])), np.empty((0, 2)))
assert len(training_history['custom train']) == 60
assert len(mlp_losses['NumPy']) == 200 and mlp_losses['NumPy'][-1] < mlp_losses['NumPy'][0]
assert all(np.isfinite(values).all() for values in training_history.values())
assert len(graph_nodes(graph_loss)) == 4
assert np.array_equal(np.sort(np.concatenate((train_idx, val_idx, test_idx))), np.arange(342))
torch.testing.assert_close(mean, raw_X[train_idx].mean(dim=0))
torch.testing.assert_close(scale, raw_X[train_idx].std(dim=0, correction=0))
print('PASS autodiff regression rejection, supplied primitives and data isolation')
'''


def verify_autodiff(execute, write_outputs=False):
    instructor = nbformat.read(SOLUTIONS, as_version=4)
    nbformat.validate(instructor)
    generated = build_autodiff()
    actual = clear_execution(json.loads(STUDENT.read_text()))
    assert actual == generated, "Rebuild the autodiff student notebook"
    student = nbformat.reads(json.dumps(generated), as_version=4)
    nbformat.validate(student)
    assert not any("reference" in c.metadata.get("tags", []) for c in student.cells)
    assert not any(c.outputs for c in student.cells if c.cell_type == "code")
    instructor_by_id = {cell.id: cell for cell in instructor.cells}
    student_by_id = {cell.id: cell for cell in student.cells}
    for cell_id, reference_id in EXERCISES.items():
        assert "raise NotImplementedError" in student_by_id[cell_id].source
        assert reference_id not in student_by_id
    print("PASS generated autodiff student source and reference removal", flush=True)

    # Unfilled path must name the first missing activity, without a solution fallback.
    try:
        execute(copy.deepcopy(student))
    except CellExecutionError as error:
        assert "NotImplementedError" in str(error) and "Complete Activity 1" in str(error)
    else:
        raise AssertionError("The unfilled student notebook did not stop at Activity 1")
    print("PASS unfilled autodiff student path stops at Activity 1", flush=True)

    executed_instructor = execute(instructor)
    print("PASS clean autodiff instructor execution, including optional sections", flush=True)
    completed_student = copy.deepcopy(student)
    for cell in completed_student.cells:
        if cell.id in EXERCISES:
            cell.source = instructor_by_id[EXERCISES[cell.id]].source
    completed_student.cells.append(nbformat.v4.new_code_cell(REGRESSION_CHECKS,
                                                            id="autodiff-private-verification"))
    execute(completed_student)
    print("PASS completed autodiff student path and rejection of fourteen legacy-style errors", flush=True)

    if write_outputs:
        executed_instructor.metadata.kernelspec = {
            "name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(executed_instructor, SOLUTIONS)
        html, _ = HTMLExporter().from_notebook_node(executed_instructor)
        SOLUTIONS.with_suffix(".html").write_text(html)
        final_student = build_autodiff()
        STUDENT.write_text(json.dumps(final_student, indent=1, ensure_ascii=False) + "\n")
        html, _ = HTMLExporter().from_notebook_node(nbformat.reads(json.dumps(final_student), as_version=4))
        STUDENT.with_suffix(".html").write_text(html)
