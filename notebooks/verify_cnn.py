"""PT04 execution, student export and regression checks for verify_notebooks.py."""
import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from make_student_notebook import build_pt04, clear_execution

ROOT = Path(__file__).resolve().parent
NAME = "PT04_Convolutional_neural_networks"
SOLUTIONS = ROOT / f"{NAME}_solutions.ipynb"
STUDENT = ROOT / f"{NAME}.ipynb"
EXERCISES = {f"cnn-{name}-exercise": f"cnn-{name}-reference" for name in [
    "conv", "pool", "depthwise", "model", "equivariance", "occlusion", "bn", "fuse",
    "detective", "residual", "dropout", "opt-local", "opt-strided", "opt-rf",
    "opt-causal", "opt-mixup"]}
# The completed student path checks the code, not the training curves: shorter runs.
QUICK_RUNS = {"EPOCHS = 4": "EPOCHS = 1", "DEPTHS, SUBSET = [2, 8, 16], 10_000": "DEPTHS, SUBSET = [2], 2_000",
              "REGULARIZATION_EPOCHS = 6": "REGULARIZATION_EPOCHS = 1"}


REGRESSION_CHECKS = r'''
# Private verification: every activity check must reject plausible wrong code.
def must_reject(check, label):
    try:
        check()
    except (AssertionError, RuntimeError, ValueError):
        return
    raise AssertionError(f'The checks accepted an incorrect implementation: {label}')

def swap(name, replacement, check, label):
    saved = globals()[name]
    globals()[name] = replacement
    try:
        must_reject(check, label)
    finally:
        globals()[name] = saved

GoodConv = MyConv2d
class PositionsAsChannels(GoodConv):
    def forward(self, xb):
        B, _, H, W = xb.shape
        s, p, st = self.kernel_size, self.padding, self.stride
        Ho, Wo = (H + 2 * p - s) // st + 1, (W + 2 * p - s) // st + 1
        Z = self.weight.view(self.weight.shape[0], -1) @ F.unfold(xb, s, padding=p, stride=st) + self.bias[:, None]
        return Z.reshape(B, Ho, Wo, -1).permute(0, 3, 1, 2)
class ChannelsLastFilters(GoodConv):
    def forward(self, xb):
        B, _, H, W = xb.shape
        s, p, st = self.kernel_size, self.padding, self.stride
        Ho, Wo = (H + 2 * p - s) // st + 1, (W + 2 * p - s) // st + 1
        flat = self.weight.permute(0, 2, 3, 1).reshape(self.weight.shape[0], -1)
        return (flat @ F.unfold(xb, s, padding=p, stride=st) + self.bias[:, None]).view(B, -1, Ho, Wo)
class SmallInit(GoodConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.weight.mul_(math.sqrt(0.5))
class NoPadding(GoodConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
for wrong in [PositionsAsChannels, ChannelsLastFilters, SmallInit, NoPadding]:
    swap('MyConv2d', wrong, check_conv, wrong.__name__)

class OverlappingPool(MyMaxPool2d):
    def forward(self, xb):
        return F.max_pool2d(xb, self.kernel_size, stride=1)
class MixedChannelsPool(MyMaxPool2d):
    def forward(self, xb):
        B, C, H, W = xb.shape
        k = self.kernel_size
        return F.unfold(xb, k, stride=k).view(B, k * k, C, -1).max(dim=1).values.view(B, C, H // k, W // k)
class AveragePool(MyMaxPool2d):
    def forward(self, xb):
        return F.avg_pool2d(xb, self.kernel_size)
for wrong in [OverlappingPool, MixedChannelsPool, AveragePool]:
    swap('MyMaxPool2d', wrong, check_pool, wrong.__name__)
swap('global_avg_pool', lambda x: x.amax(dim=(2, 3)), check_pool, 'global max pooling')

class MixedDepthwise(MyDepthwiseConv2d):
    def forward(self, xb):
        B, C, H, W = xb.shape
        s, p = self.kernel_size, self.padding
        patches = F.unfold(xb, s, padding=p).view(B, s * s, C, -1).transpose(1, 2)
        Z = (patches * self.weight.view(C, s * s, 1)).sum(dim=2) + self.bias[:, None]
        return Z.view(B, C, H + 2 * p - s + 1, W + 2 * p - s + 1)
swap('MyDepthwiseConv2d', MixedDepthwise, check_depthwise, 'depthwise with mixed channel order')

class TorchLayersCNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.head = MyLinear(8, n_classes)
    def forward(self, xb):
        return self.head(global_avg_pool(F.relu(self.conv(xb))))
class LargeCNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = MyConv2d(1, 32, 3, padding=1)  # About 251,000 parameters.
        self.head = MyLinear(32 * 28 * 28, n_classes)
    def forward(self, xb):
        return self.head(F.relu(self.conv(xb)).flatten(1))
for wrong in [TorchLayersCNN, LargeCNN]:
    swap('MyCNN', wrong, check_cnn, wrong.__name__)

@torch.no_grad()
def uncropped(layer, x, dy, dx, border):
    return (layer(shift(x, dy, dx)) - shift(layer(x), dy, dx)).abs().max().item()
@torch.no_grad()
def output_not_shifted(layer, x, dy, dx, border):
    difference = layer(shift(x, dy, dx)) - layer(x)
    return difference[..., border:-border, border:-border].abs().max().item() if border else difference.abs().max().item()
for wrong in [uncropped, output_not_shifted]:
    swap('equivariance_error', wrong, check_equivariance_error, wrong.__name__)

good_occlusion = occlusion_map
@torch.no_grad()
def transposed_occlusion(model, image, label, patch=7, stride=3):
    return good_occlusion(model, image, label, patch, stride).T
@torch.no_grad()
def black_occlusion(model, image, label, patch=7, stride=3):
    model.eval()
    H, W = image.shape[-2:]
    rows, cols = range(0, H - patch + 1, stride), range(0, W - patch + 1, stride)
    batch = image.repeat(len(rows) * len(cols), 1, 1, 1)
    for n, (i, j) in enumerate((i, j) for i in rows for j in cols):
        batch[n, :, i:i + patch, j:j + patch] = 0.
    return model(batch).softmax(dim=1)[:, label].view(len(rows), len(cols))
for wrong in [transposed_occlusion, black_occlusion]:
    swap('occlusion_map', wrong, check_occlusion, wrong.__name__)

GoodBN = MyBatchNorm2d
class BiasedRunningVar(GoodBN):
    def forward(self, xb):
        if self.training:
            mean, var = xb.mean(dim=(0, 2, 3)), xb.var(dim=(0, 2, 3), correction=0)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return self.weight[:, None, None] * (xb - mean[:, None, None]) / torch.sqrt(var[:, None, None] + self.eps) + self.bias[:, None, None]
class FlippedMomentum(GoodBN):
    def forward(self, xb):
        if self.training:
            mean, var = xb.mean(dim=(0, 2, 3)), xb.var(dim=(0, 2, 3), correction=0)
            n = xb.numel() / xb.shape[1]
            with torch.no_grad():
                self.running_mean.lerp_(mean, 1 - self.momentum)
                self.running_var.lerp_(var * n / (n - 1), 1 - self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return self.weight[:, None, None] * (xb - mean[:, None, None]) / torch.sqrt(var[:, None, None] + self.eps) + self.bias[:, None, None]
class BatchStatsAtInference(GoodBN):
    def forward(self, xb):
        was = self.training
        self.training = True
        try:
            return super().forward(xb)
        finally:
            self.training = was
class PerPixelStats(GoodBN):
    def forward(self, xb):
        mean, var = xb.mean(dim=0, keepdim=True), xb.var(dim=0, correction=0, keepdim=True)
        return self.weight[:, None, None] * (xb - mean) / torch.sqrt(var + self.eps) + self.bias[:, None, None]
for wrong in [BiasedRunningVar, FlippedMomentum, BatchStatsAtInference, PerPixelStats]:
    swap('MyBatchNorm2d', wrong, check_batchnorm, wrong.__name__)

good_fuse = fuse_conv_bn
def fuse_without_conv_bias(conv, bn):
    fused = copy.deepcopy(conv)
    with torch.no_grad():
        a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.copy_(conv.weight * a[:, None, None, None])
        fused.bias.copy_(bn.bias - a * bn.running_mean)
    return fused
def fuse_in_place(conv, bn):
    fused = good_fuse(conv, bn)
    with torch.no_grad():
        conv.weight.copy_(fused.weight)
        conv.bias.copy_(fused.bias)
    return fused
for wrong in [fuse_without_conv_bias, fuse_in_place]:
    swap('fuse_conv_bn', wrong, check_fusion, wrong.__name__)

@torch.no_grad()
def forgets_mode(model, x):
    model.eval()
    return model(x).softmax(dim=1)
@torch.no_grad()
def no_buffer_check(model, x):
    was = model.training
    model.eval()
    out = model(x).softmax(dim=1)
    model.train(was)
    return out
@torch.no_grad()
def current_mode(model, x):
    return model(x).softmax(dim=1)
for wrong in [forgets_mode, no_buffer_check, current_mode]:
    swap('safe_evaluate', wrong, check_safe_evaluate, wrong.__name__)

GoodBlock = ResidualBlock
class ReluBeforeSum(GoodBlock):
    def forward(self, xb):
        h = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(xb)))))
        return F.relu(h) + xb if self.residual else F.relu(h)
class BiasWithBN(GoodBlock):
    def __init__(self, channels, residual=True, batchnorm=True):
        super().__init__(channels, residual, batchnorm)
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
for wrong in [ReluBeforeSum, BiasWithBN]:
    swap('ResidualBlock', wrong, check_residual, wrong.__name__)

class Unscaled(MyDropout):
    def forward(self, xb):
        return xb * (torch.rand_like(xb) > self.p) if self.training else xb
class FixedMask(MyDropout):
    def forward(self, xb):
        if not self.training:
            return xb
        mask = torch.rand(xb.shape, generator=torch.Generator().manual_seed(0)) > self.p
        return xb * mask / (1 - self.p)
class AlwaysOn(MyDropout):
    def forward(self, xb):
        return xb * (torch.rand_like(xb) > self.p) / (1 - self.p)
for wrong in [Unscaled, FixedMask, AlwaysOn]:
    swap('MyDropout', wrong, check_dropout, wrong.__name__)

# Supplied data handling: disjoint splits and statistics from the training images only.
assert not set(train_idx.tolist()) & set(val_idx.tolist())
assert len(train_data) == 54_000 and len(val_data) == 6_000
assert abs(mean - train_raw.data[train_idx].float().div(255).mean().item()) < 1e-6
assert abs(background - (0 - mean) / std) < 1e-6
print('PASS PT04 checks reject 29 incorrect implementations; data splits and statistics are isolated')
'''


def verify_cnn(write_outputs=False):
    def execute(nb):
        return NotebookClient(nb, timeout=900, kernel_name="nnds-validation",
                              resources={"metadata": {"path": str(ROOT)}}).execute()

    instructor = nbformat.read(SOLUTIONS, as_version=4)
    nbformat.validate(instructor)
    generated = build_pt04()
    actual = clear_execution(json.loads(STUDENT.read_text()))
    assert actual == generated, "Rebuild the PT04 student notebook"
    student = nbformat.reads(json.dumps(generated), as_version=4)
    nbformat.validate(student)
    assert not any("reference" in c.metadata.get("tags", []) for c in student.cells)
    assert not any(c.outputs for c in student.cells if c.cell_type == "code")
    instructor_by_id = {cell.id: cell for cell in instructor.cells}
    student_by_id = {cell.id: cell for cell in student.cells}
    for cell_id, reference_id in EXERCISES.items():
        assert "raise NotImplementedError" in student_by_id[cell_id].source, cell_id
        assert reference_id not in student_by_id
    print("PASS generated PT04 student source and reference removal", flush=True)

    try:
        execute(copy.deepcopy(student))
    except CellExecutionError as error:
        assert "NotImplementedError" in str(error) and "Complete Activity 1" in str(error)
    else:
        raise AssertionError("The unfilled PT04 student notebook did not stop at Activity 1")
    print("PASS unfilled PT04 student path stops at Activity 1", flush=True)

    executed_instructor = execute(instructor)
    print("PASS clean PT04 instructor execution, including optional sections", flush=True)

    completed = copy.deepcopy(student)
    for cell in completed.cells:
        if cell.id in EXERCISES:
            cell.source = instructor_by_id[EXERCISES[cell.id]].source
        for full, quick in QUICK_RUNS.items():
            cell.source = cell.source.replace(full, quick)
    sources = "\n".join(c.source for c in completed.cells)
    assert not any(f"\n{full}" in sources or sources.startswith(full) for full in QUICK_RUNS), "A training setting was not shortened"
    assert all(quick in sources for quick in QUICK_RUNS.values())
    completed.cells.append(nbformat.v4.new_code_cell(REGRESSION_CHECKS, id="cnn-private-verification"))
    execute(completed)
    print("PASS completed PT04 student path and rejection of incorrect implementations", flush=True)

    if write_outputs:
        executed_instructor.metadata.kernelspec = {
            "name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(executed_instructor, SOLUTIONS)
        html, _ = HTMLExporter().from_notebook_node(executed_instructor)
        SOLUTIONS.with_suffix(".html").write_text(html)
        final_student = build_pt04()
        STUDENT.write_text(json.dumps(final_student, indent=1, ensure_ascii=False) + "\n")
        html, _ = HTMLExporter().from_notebook_node(nbformat.reads(json.dumps(final_student), as_version=4))
        STUDENT.with_suffix(".html").write_text(html)
