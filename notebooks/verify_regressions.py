"""Private checks run by verify_notebooks.py; they are never distributed.

REGRESSIONS[lab] runs at the end of the completed student notebook: every activity
check must reject plausible wrong implementations, and the supplied code must keep
the data splits isolated. INSTRUCTOR_CHECKS[lab] runs at the end of the instructor
notebook. extra_checks builds small notebooks from instructor cells (PT01 only).
"""
import nbformat

COMMON = r'''
def must_reject(check, label):
    try:
        check()
    except (AssertionError, RuntimeError, ValueError, TypeError):
        return
    raise AssertionError(f'The checks accepted an incorrect implementation: {label}')
'''

PT01 = COMMON + r'''
must_reject(lambda: check_sum_diagonals(lambda A: 2 * A.diagonal().sum()), 'main diagonal counted twice')
must_reject(lambda: check_normalize_rows(
    lambda A, eps=1e-8: (A - A.mean(dim=0, keepdim=True)) / A.std(dim=0, correction=0, keepdim=True).clamp_min(eps)),
    'normalization over the wrong axis')
must_reject(lambda: check_normalize_rows(
    lambda A, eps=1e-8: (A - A.mean(dim=1, keepdim=True)) / A.std(dim=1, correction=0, keepdim=True)),
    'nan for a constant row')
must_reject(lambda: check_cosine_similarity(lambda A: A @ A.T), 'unnormalized dot products')
print('PASS PT01 array checks reject wrong implementations')
'''

PT02 = COMMON + r'''
from torch.utils.data import DataLoader, TensorDataset
# Split isolation and train-only preprocessing.
assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))
assert set(train_idx) | set(val_idx) | set(test_idx) == set(range(len(y)))
torch.testing.assert_close(mean, X[train_idx].mean(dim=0))
torch.testing.assert_close(scale, X[train_idx].std(dim=0, correction=0))
assert history['train_loss'][-1] < history['train_loss'][0]
assert test_accuracy > baseline_accuracy
assert batch_sizes == [32] * 6 + [13] and mini_updates == 210
assert selected_lr == min(learning_rates, key=lambda r: lr_histories[r]['val_loss'][-1])
assert model is lr_models[selected_lr]
# The optional diagnostic preserves the data and the trained classifier.
torch.testing.assert_close(analysis_y, ytrain)
for diagnostic_parameter, saved_parameter in zip(analysis_model.parameters(), lr_models[0.1].parameters()):
    torch.testing.assert_close(diagnostic_parameter, saved_parameter.double())
assert per_example_loop.shape == (len(ytrain), 15)
torch.testing.assert_close(per_example_transformed.mean(dim=0), mean_gradient)
assert torch.isnan(gradient_alignment(torch.zeros_like(per_example_loop[:1]), mean_gradient)).all()

must_reject(lambda: check_linear_logits(lambda X, W, b: X @ W.T), 'missing bias')
must_reject(lambda: check_linear_logits(lambda X, W, b: X @ W.T + b[0]), 'one bias for every class')
must_reject(lambda: check_cross_entropy(lambda z, t: F.cross_entropy(z, t, reduction='sum')), 'summed loss')
must_reject(lambda: check_cross_entropy(
    lambda z, t: -z.softmax(dim=1)[torch.arange(len(t)), t].log().mean()), 'log of softmax overflows')
must_reject(lambda: check_cross_entropy(
    lambda z, t: (torch.logsumexp(z, dim=1) - z[:, t]).mean()), 'wrong selection of the target logits')
must_reject(lambda: check_training_step(suspect_step), 'gradients never cleared')
def loss_after_update(model, X, y, lr):
    training_step(model, X, y, lr)
    return cross_entropy(model(X), y).item()
must_reject(lambda: check_training_step(loss_after_update), 'loss computed after the update')
def tensor_loss(model, X, y, lr):
    return torch.tensor(training_step(model, X, y, lr))
must_reject(lambda: check_training_step(tensor_loss), 'loss returned as a tensor')
def frozen_bias(model, X, y, lr):
    saved = model.bias.detach().clone()
    loss = training_step(model, X, y, lr)
    with torch.no_grad():
        model.bias.copy_(saved)
    return loss
must_reject(lambda: check_training_step(frozen_bias), 'bias not updated')
print('PASS PT02 checks reject wrong implementations; splits and diagnostics are isolated')
'''

PT02_INSTRUCTOR = r'''
assert len(image_runs) == 4
assert not (image_train_mask & image_heldout_mask).any() and (image_train_mask | image_heldout_mask).all()
for (encoding, width), (coordinate_model, trace, snapshots) in image_runs.items():
    assert set(snapshots) == {0, 100, 400}
    assert trace['update'][0] == 0 and trace['update'][-1] == 400
    assert trace['train_mse'][-1] < trace['train_mse'][0]
    assert all(np.isfinite(trace[key]).all() for key in ['train_mse', 'heldout_mse'])
    assert 'frequencies' not in dict(coordinate_model.named_parameters())
    assert 'frequencies' in coordinate_model.state_dict()
    output = coordinate_model(coordinate_grid(64))
    assert output.shape == (4096, 3) and ((output >= 0) & (output <= 1)).all()
assert image_summary['float32 payload bytes'].tolist() == [1484, 5004, 3532, 8972]
assert model is lr_models[selected_lr]
'''

PT03 = COMMON + r'''
def expect_error(kind, function):
    try:
        function()
    except kind:
        return
    raise AssertionError(f'Expected {kind.__name__}')

must_reject(lambda: check_elementwise_vjps(add_vjp, lambda a, b, v: (v * a, v * b)), 'swapped multiplication factors')
must_reject(lambda: check_elementwise_vjps(lambda a, b, v: (v, 0 * v), multiply_vjp), 'addition ignores b')
must_reject(lambda: check_matmul_vjp(lambda a, b, v: (v @ b.T, a.T @ np.ones_like(v))), 'matmul ignores v for b')
must_reject(lambda: check_matmul_vjp(lambda a, b, v: (v @ b, a.T @ v)), 'missing transpose')
must_reject(lambda: check_nonlinear_vjps(lambda y, v: v / np.cosh(y) ** 2, relu_vjp), 'tanh derivative of the output')
must_reject(lambda: check_nonlinear_vjps(tanh_vjp, lambda x, v: (x >= 0) * v), 'relu passes the adjoint at 0')

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
def resetting_backward(self, gradient=None):
    for node in topological_order(self):
        node.grad = None
    saved_backward(self, gradient)
def retaining_backward(self, gradient=None):
    saved_backward(self, gradient)
    for node in topological_order(self):
        if node.vjp is not None:
            node.grad = np.zeros_like(node.data)
for wrong, label in [(overwriting_backward, 'overwritten adjoints'),
                     (lambda self, gradient=None: premature_backward(self), 'premature traversal'),
                     (resetting_backward, 'no accumulation'), (retaining_backward, 'intermediate .grad')]:
    Tensor.backward = wrong
    try:
        must_reject(check_backward, label)
    finally:
        Tensor.backward = saved_backward

def frozen_bias(X, targets, W, b, lr):
    result = classifier_step(X, targets, W, b, lr)
    result['b'] = b.copy()
    return result
must_reject(lambda: check_classifier_step(frozen_bias), 'frozen bias')
def in_place(X, targets, W, b, lr):
    result = classifier_step(X, targets, W, b, lr)
    W -= lr * result['dW']
    return result
must_reject(lambda: check_classifier_step(in_place), 'W modified in place')
saved_ce = cross_entropy
cross_entropy = lambda logits, targets: saved_ce(logits, targets) * len(targets)
try:
    must_reject(lambda: check_classifier_step(classifier_step), 'summed loss')
finally:
    cross_entropy = saved_ce

# The supplied matmul turns vectors into matrices; compare all four cases with PyTorch.
gen = np.random.default_rng(91)
for a_shape, b_shape in [((4,), (4,)), ((3, 4), (4,)), ((4,), (4, 3)), ((2, 4), (4, 3))]:
    a = Tensor(gen.normal(size=a_shape), requires_grad=True)
    b = Tensor(gen.normal(size=b_shape), requires_grad=True)
    out = a @ b
    seed = gen.normal(size=out.data.shape)
    out.backward(seed)
    ta, tb = torch.tensor(a.data, requires_grad=True), torch.tensor(b.data, requires_grad=True)
    (ta @ tb).backward(torch.tensor(seed))
    close(a.grad, ta.grad.numpy(), f'matmul {a_shape} @ {b_shape}: a.grad')
    close(b.grad, tb.grad.numpy(), f'matmul {a_shape} @ {b_shape}: b.grad')
# Broadcasting through sum_to_shape, against PyTorch.
for a_shape, b_shape in [((3, 3), (3,)), ((2, 1, 4), (1, 3, 1)), ((), ()), ((4, 1), (1, 3))]:
    a = Tensor(gen.normal(size=a_shape), requires_grad=True)
    b = Tensor(gen.normal(size=b_shape), requires_grad=True)
    (a * b + a).sum().backward()
    ta, tb = torch.tensor(a.data, requires_grad=True), torch.tensor(b.data, requires_grad=True)
    ((ta * tb) + ta).sum().backward()
    close(a.grad, ta.grad.numpy())
    close(b.grad, tb.grad.numpy())
# Supplied primitives: seeds, validation, copies.
z = Tensor([[1000., 1001., -999.], [10., 11., 12.]], requires_grad=True)
t = np.array([0, 2])
cross_entropy(z, t).backward(np.array(3.))
tripled = z.grad
z.grad = None
cross_entropy(z, t).backward()
close(tripled, 3 * z.grad)
expect_error(ValueError, lambda: cross_entropy(z, np.array([0., 1.])))
expect_error(ValueError, lambda: Tensor(np.ones((2, 2, 2))) @ Tensor([1., 2.]))
expect_error(ValueError, lambda: sum_to_shape(np.ones((2, 3)), (2,)))
expect_error(RuntimeError, lambda: Tensor([1., 2.], requires_grad=True).backward())
close(sum_to_shape(np.ones((2, 3, 4)), (1, 3, 1)), np.full((1, 3, 1), 8.))
cube = Tensor(np.arange(24.).reshape(2, 3, 4), requires_grad=True)
cube.sum(axis=(0, -1)).backward(np.array([1., 2., 3.]))
close(cube.grad, np.broadcast_to(np.array([1., 2., 3.])[None, :, None], (2, 3, 4)))
# Twelve shared additions: each VJP is called once (13 nodes, not 4096 paths); a deep chain.
leaf = Tensor(1., requires_grad=True)
repeated, counts = leaf, {}
for i in range(12):
    def counted(v, key=i):
        counts[key] = counts.get(key, 0) + 1
        return (v, v)
    repeated = Tensor(2 * repeated.data, parents=(repeated, repeated), vjp=counted, name='+')
repeated.backward()
close(leaf.grad, 4096.)
assert counts == {i: 1 for i in range(12)}
deep = Tensor(0., requires_grad=True)
for i in range(1500):
    deep = Tensor(deep.data + 1, parents=(deep,), vjp=lambda v: (v,), name='increment')
deep.backward()
close(grad(lambda x: x * x)(np.array(3.)), np.array(6.))
close(jacobian(lambda x: Tensor(np.array([])), np.array([1., 2.])), np.empty((0, 2)))
assert len(training_history['custom train']) == 60
assert len(mlp_losses['NumPy']) == 200 and mlp_losses['NumPy'][-1] < mlp_losses['NumPy'][0]
assert np.array_equal(np.sort(np.concatenate((train_idx, val_idx, test_idx))), np.arange(342))
torch.testing.assert_close(mean, raw_X[train_idx].mean(dim=0))
print('PASS PT03 checks reject wrong implementations; supplied primitives and data isolation')
'''

PT04 = COMMON + r'''
GoodConv = MyConv2d
class PositionsAsChannels(GoodConv):
    def forward(self, xb):
        B, _, H, W = xb.shape
        s, p = self.kernel_size, self.padding
        Z = self.weight.view(self.weight.shape[0], -1) @ F.unfold(xb, s, padding=p) + self.bias[:, None]
        return Z.reshape(B, H + 2 * p - s + 1, W + 2 * p - s + 1, -1).permute(0, 3, 1, 2)
class ChannelsLastFilters(GoodConv):
    def forward(self, xb):
        B, _, H, W = xb.shape
        s, p = self.kernel_size, self.padding
        flat = self.weight.permute(0, 2, 3, 1).reshape(self.weight.shape[0], -1)
        return (flat @ F.unfold(xb, s, padding=p) + self.bias[:, None]).view(B, -1, H + 2 * p - s + 1, W + 2 * p - s + 1)
class NoPadding(GoodConv):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
class NoBias(GoodConv):
    def forward(self, xb):
        return super().forward(xb) - self.bias[:, None, None]
class PlainTensorWeight(GoodConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight = self.weight.detach().clone()
        del self.weight
        self.weight = weight
for wrong in [PositionsAsChannels, ChannelsLastFilters, NoPadding, NoBias, PlainTensorWeight]:
    must_reject(lambda: check_conv(wrong), wrong.__name__)

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
    must_reject(lambda: check_pool(wrong), wrong.__name__)

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
    must_reject(lambda: check_cnn(wrong), wrong.__name__)

@torch.no_grad()
def transposed_occlusion(model, image, label, patch=7, stride=3):
    return occlusion_map(model, image, label, patch, stride).T
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
    must_reject(lambda: check_occlusion(wrong, cnn), wrong.__name__)

class IgnoresStride(MyStridedConv2d):
    def forward(self, xb):
        return MyConv2d.forward(self, xb)
must_reject(lambda: check_strided_conv(IgnoresStride), 'stride ignored')

class MixedDepthwise(MyDepthwiseConv2d):
    def forward(self, xb):
        B, C, H, W = xb.shape
        s, p = self.kernel_size, self.padding
        patches = F.unfold(xb, s, padding=p).view(B, s * s, C, -1).transpose(1, 2)
        Z = (patches * self.weight.view(C, s * s, 1)).sum(dim=2) + self.bias[:, None]
        return Z.view(B, C, H + 2 * p - s + 1, W + 2 * p - s + 1)
must_reject(lambda: check_depthwise(MixedDepthwise), 'depthwise with mixed channel order')

@torch.no_grad()
def uncropped(layer, x, dy, dx, border):
    return (layer(shift(x, dy, dx)) - shift(layer(x), dy, dx)).abs().max().item()
@torch.no_grad()
def output_not_shifted(layer, x, dy, dx, border):
    difference = layer(shift(x, dy, dx)) - layer(x)
    return difference[..., border:-border, border:-border].abs().max().item() if border else difference.abs().max().item()
for wrong in [uncropped, output_not_shifted]:
    must_reject(lambda: check_equivariance_error(wrong), wrong.__name__)

must_reject(lambda: check_causal_conv1d(lambda x, w, b: F.conv1d(x, w, b, padding=2)), 'centered convolution')
must_reject(lambda: check_receptive_field(lambda network, size=64: (1, 1)), 'constant receptive field')

# Supplied data handling: disjoint splits and statistics from the training images only.
assert not set(train_idx.tolist()) & set(val_idx.tolist())
assert len(train_data) == 54_000 and len(val_data) == 6_000
assert abs(mean - train_raw.data[train_idx].float().div(255).mean().item()) < 1e-6
print('PASS PT04 checks reject wrong implementations; data splits and statistics are isolated')
'''

PT05 = COMMON + r'''
GoodBN = MyBatchNorm2d
def bn_forward(self, xb, mean, var):
    return self.weight[:, None, None] * (xb - mean[:, None, None]) / torch.sqrt(var[:, None, None] + self.eps) + self.bias[:, None, None]
class BiasedRunningVar(GoodBN):  # Accepted: the check allows the biased running variance.
    def forward(self, xb):
        if self.training:
            mean, var = xb.mean(dim=(0, 2, 3)), xb.var(dim=(0, 2, 3), correction=0)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return bn_forward(self, xb, mean, var)
check_batchnorm(BiasedRunningVar)
class FlippedMomentum(GoodBN):
    def forward(self, xb):
        if self.training:
            mean, var = xb.mean(dim=(0, 2, 3)), xb.var(dim=(0, 2, 3), correction=0)
            with torch.no_grad():
                self.running_mean.lerp_(mean, 1 - self.momentum)
                self.running_var.lerp_(var, 1 - self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return bn_forward(self, xb, mean, var)
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
class NoRunningUpdate(GoodBN):
    def forward(self, xb):
        mean, var = (xb.mean(dim=(0, 2, 3)), xb.var(dim=(0, 2, 3), correction=0)) if self.training else (self.running_mean, self.running_var)
        return bn_forward(self, xb, mean, var)
class TrackedRunningStats(GoodBN):
    def forward(self, xb):
        if self.training:
            mean, var = xb.mean(dim=(0, 2, 3)), xb.var(dim=(0, 2, 3), correction=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        return bn_forward(self, xb, mean, var)
class ParametersAsBuffers(GoodBN):
    def __init__(self, channels, eps=1e-5, momentum=0.1):
        super().__init__(channels, eps, momentum)
        del self.weight, self.bias
        self.register_buffer('weight', torch.ones(channels))
        self.register_buffer('bias', torch.zeros(channels))
for wrong in [FlippedMomentum, BatchStatsAtInference, PerPixelStats, NoRunningUpdate, TrackedRunningStats, ParametersAsBuffers]:
    must_reject(lambda: check_batchnorm(wrong), wrong.__name__)

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
    must_reject(lambda: check_residual(wrong), wrong.__name__)

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
    must_reject(lambda: check_dropout(wrong), wrong.__name__)

def fuse_without_conv_bias(conv, bn):
    fused = copy.deepcopy(conv)
    with torch.no_grad():
        a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.copy_(conv.weight * a[:, None, None, None])
        fused.bias.copy_(bn.bias - a * bn.running_mean)
    return fused
def fuse_in_place(conv, bn):
    fused = fuse_conv_bn(conv, bn)
    with torch.no_grad():
        conv.weight.copy_(fused.weight)
        conv.bias.copy_(fused.bias)
    return fused
for wrong in [fuse_without_conv_bias, fuse_in_place]:
    must_reject(lambda: check_fusion(wrong), wrong.__name__)

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
    must_reject(lambda: check_safe_evaluate(wrong, model_bn), wrong.__name__)

def mixup_per_example(x, y, alpha=0.4, n_classes=10):
    lam = torch.distributions.Beta(alpha, alpha).sample((len(x),))
    perm = torch.randperm(len(x))
    targets = F.one_hot(y, n_classes).float()
    return lam[:, None, None, None] * x + (1 - lam[:, None, None, None]) * x[perm], lam[:, None] * targets + (1 - lam[:, None]) * targets[perm]
must_reject(lambda: check_augment_mixup(augment, mixup_per_example), 'one lambda per example')

assert not set(train_idx.tolist()) & set(val_idx.tolist())
assert len(train_data) == 54_000 and len(val_data) == 6_000
assert abs(mean - train_raw.data[train_idx].float().div(255).mean().item()) < 1e-6
print('PASS PT05 checks reject wrong implementations; data splits and statistics are isolated')
'''

REGRESSIONS = {
    "PT01_Introduction_to_PyTorch": PT01,
    "PT02_Logistic_regression": PT02,
    "PT03_Automatic_differentiation": PT03,
    "PT04_Convolutional_neural_networks": PT04,
    "PT05_Training_deeper_networks": PT05,
}
INSTRUCTOR_CHECKS = {"PT02_Logistic_regression": PT02_INSTRUCTOR}

PT01_TENSOR_REJECTIONS = r'''
def must_reject(actual, expected):
    try:
        torch.testing.assert_close(actual, expected)
    except AssertionError:
        return
    raise AssertionError("The check accepted the original tensor-semantics bug")

must_reject(suspect_errors, torch.tensor([1., 0., -1.]))
must_reject(suspect_mse, torch.tensor(2. / 3.))
assert suspect_predictions.shape == row_predictions.shape
must_reject(suspect_predictions, torch.tensor([21., 43.]))
assert suspect_probabilities.shape == probabilities.shape
must_reject(suspect_probabilities.sum(dim=-1), torch.ones(3))
must_reject(suspect_probabilities.argmax(dim=-1), torch.tensor([0, 2, 0]))
try:
    larger_batch.T @ weights
except RuntimeError:
    pass
else:
    raise AssertionError("The transposed three-example batch should fail")
single_error = predictions[:1].squeeze(-1) - targets[:1]
assert single_error.shape == (1,)
'''

PT01_DETECTIVE_ORIGINALS = r'''
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    assert hidden.grad is None
torch.testing.assert_close(leaf.grad, torch.tensor([18., 36.]))
assert restart.is_leaf and restart.grad_fn is None and original.grad is None
torch.testing.assert_close(restart.grad, torch.tensor([6., 12.]))
torch.testing.assert_close(parameter.grad, torch.tensor(28. / 3.))

def must_raise_runtime_error(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("Expected autograd to reject this operation")

must_raise_runtime_error(stale_loss.backward)
alias_source = torch.tensor([1., 2.], requires_grad=True)
alias_loss = alias_source.square().sum()
alias_source.detach().add_(1.)
must_raise_runtime_error(alias_loss.backward)
unequal_parameter = torch.tensor(1., requires_grad=True)
for values in microbatches:
    ((unequal_parameter * values).square().mean() / len(microbatches)).backward()
torch.testing.assert_close(unequal_parameter.grad, torch.tensor(11.5))
'''


def extra_checks(stem, executed):
    """Small notebooks built from instructor cells, executed in fresh kernels."""
    if stem != "PT01_Introduction_to_PyTorch":
        return
    by_id = {cell.id: cell.source for cell in executed.cells}
    setups = [by_id[f"tensor-challenge-{i}-code"] for i in range(1, 4)]
    snippets = [by_id[f"tensor-challenge-solutions-code-{i}"] for i in range(1, 4)]
    yield "tensor-challenge solutions reject the three original bugs", nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("import torch"),
        nbformat.v4.new_code_cell("\n\n".join(setups + snippets)),
        nbformat.v4.new_code_cell(PT01_TENSOR_REJECTIONS)])
    cases = [by_id[f"autograd-case-{i}"] for i in range(1, 5)]
    repairs = [by_id[f"autograd-detective-solutions-code-{i}"] for i in range(1, 5)]
    yield "autograd detective cases behave as diagnosed, and the repairs work", nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("import torch"),
        nbformat.v4.new_code_cell("\n\n".join(cases)),
        nbformat.v4.new_code_cell(PT01_DETECTIVE_ORIGINALS),
        nbformat.v4.new_code_cell("\n\n".join(repairs))])
