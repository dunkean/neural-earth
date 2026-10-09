"""Offline base-only quantization preparation. Never imported by production.

Q/DQ is an interchange description, not evidence of integer GPU execution.
The module deliberately exposes no production inference/activation API.
"""
from __future__ import annotations

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import copy
import hashlib
import json
import platform
from pathlib import Path
import time
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_UPSTREAM = str(REPO_ROOT / 'terrain-diffusion')
if _UPSTREAM not in sys.path:
    sys.path.insert(0, _UPSTREAM)
from terrain_diffusion.models.mp_layers import MPConv

SCHEMA = 1
ROLE = 'offline-base-preview'
DEFAULT_EXCLUDE = ('out_conv', 'attn_', 'emb_linear',
                   'conditional_layers', 'noise_', 'logvar_')


def require_scope(model_kind='base', role=ROLE):
    if model_kind != 'base' or role != ROLE:
        raise ValueError('Only offline base preview preparation is supported; native/coarse/decoder refused')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def runtime_identity():
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, torch_cuda=torch.version.cuda,
                module_sha=sha_file(__file__))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


class FrozenMP(MPConv):
    """Effective MP weight with its original gain baked in on the source device."""
    def __init__(self, weight, *, groups=1, no_padding=False):
        # Remain an MPConv for EDMUnet2D.compute_embeddings' isinstance branch.
        nn.Module.__init__(self)
        self.register_buffer('weight', weight.detach().clone())
        self.groups, self.no_padding = groups, no_padding
        self.out_channels = weight.shape[0]

    def forward(self, x, gain=1):
        # All call-site gains must be frozen by freeze_base, not applied twice.
        if self.weight.ndim == 2:
            return F.linear(x, self.weight)
        return F.conv2d(x, self.weight, groups=self.groups,
                        padding=0 if self.no_padding else self.weight.shape[-1] // 2)


@torch.no_grad()
def freeze_base(model, *, model_kind='base', role=ROLE, max_weight_bytes=2 * 1024**3):
    """Copy an unwrapped, eval base. Source parameters and methods are untouched.

    Effective weights are calculated on their source device. CPU and CUDA
    normalizations are not presumed bit-identical. Caller validates forwards.
    """
    require_scope(model_kind, role)
    if model.training or hasattr(model, '_buckets'):
        raise ValueError('Expected an unwrapped eval base, not a graph wrapper')
    config = getattr(model, 'config', {})
    if (config.get('in_channels') != 5 or config.get('out_channels') not in (None, 5)
            or list(config.get('conditional_inputs', [])) not in
            ([('tensor', 58, 1.)], [['tensor', 58, 1.]])):
        raise ValueError('Expected the five-channel, condition58 base architecture')
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    if size > max_weight_bytes:
        raise ValueError('Base weight copy exceeds offline budget')
    from terrain_diffusion.models.mp_layers import MPConv, MPConvResample, normalize
    if any(isinstance(m, MPConvResample) for m in model.modules()):
        raise ValueError('Learned resampling is not supported by this base exporter')
    frozen = copy.deepcopy(model)
    specs = {}
    source_modules = dict(model.named_modules())
    for name, module in source_modules.items():
        if not isinstance(module, MPConv):
            continue
        gain = 1
        if name == 'out_conv':
            gain = model.out_gain
        elif name.endswith('.emb_linear'):
            gain = source_modules[name.rsplit('.', 1)[0]].emb_gain
        weight = normalize(module.weight.to(torch.float32))
        weight = (weight * (gain / np.sqrt(weight[0].numel()))).to(module.weight.dtype)
        replacement = FrozenMP(weight, groups=module.groups, no_padding=module.no_padding)
        parent_name, _, child_name = name.rpartition('.')
        parent = frozen.get_submodule(parent_name) if parent_name else frozen
        setattr(parent, child_name, replacement)
        specs[name] = dict(shape=list(weight.shape), dtype=str(weight.dtype), groups=module.groups,
                           no_padding=module.no_padding, sha256=tensor_sha(weight),
                           source_device=str(module.weight.device), gain_sha=tensor_sha(torch.as_tensor(gain)))
    if not specs:
        raise ValueError('No MPConv found: refusing an unrelated model')
    for module in frozen.modules():
        # Copy must not retain monitoring hooks or inherited bound terrain patches.
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        for key in list(module.__dict__):
            if key.startswith('_terrain_'):
                module.__dict__.pop(key)
    frozen.eval().requires_grad_(False)
    return frozen, specs


def calibration_positions(length, take, *, device='cpu'):
    """Inclusive sparse indices without float32 endpoint rounding above 2**24.

    Bound the integer product too, rather than allowing int64 wraparound on
    hypothetical enormous arrays. Only ``take`` elements are allocated.
    """
    if type(length) is not int or type(take) is not int or not 1 <= take <= length:
        raise ValueError('Expected integer length >= take >= 1')
    if (length-1) * max(1, take-1) > torch.iinfo(torch.int64).max:
        raise ValueError('Calibration index product exceeds int64')
    return torch.arange(take, device=device, dtype=torch.int64) * (length-1) // max(1, take-1)


class ActivationCollector:
    """Bounded real layer-input calibration, synchronous and intentionally offline."""
    def __init__(self, *, max_calls=128, max_values_per_layer=32768):
        if max_calls < 1 or max_values_per_layer < 1:
            raise ValueError('Calibration limits must be positive')
        self.max_calls, self.max_values = max_calls, max_values_per_layer
        self.stats, self.handles, self.contexts = {}, [], []
        self.active = False

    def start(self, metadata):
        self.active = len(self.contexts) < self.max_calls
        if self.active:
            self.contexts.append(copy.deepcopy(metadata))
        return self.active

    def observe(self, name, value):
        if not self.active:
            return
        x = value.detach().float().flatten()
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f'Non-finite calibration activation: {name}')
        entry = self.stats.setdefault(name, dict(calls=0, elements=0, max_abs=0., sum_squares=0., values=[]))
        entry['calls'] += 1
        entry['elements'] += x.numel()
        entry['max_abs'] = max(entry['max_abs'], float(x.abs().max()))
        entry['sum_squares'] += float(x.double().square().sum())
        remaining = self.max_values - len(entry['values'])
        if remaining:
            # Deterministic sparse diagnostic sample. Max/minmax uses ALL values.
            take = min(remaining, 1024, x.numel())
            positions = calibration_positions(x.numel(), take, device=x.device)
            entry['values'].extend(x[positions].cpu().tolist())

    def install(self, model):
        for name, module in model.named_modules():
            if isinstance(module, FrozenMP) or type(module).__name__ == 'MPConv':
                self.handles.append(module.register_forward_pre_hook(
                    lambda m, args, name=name: self.observe(name, args[0])))
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False

    def result(self):
        if not self.contexts or not self.stats:
            raise ValueError('Empty calibration is not valid')
        layers = {}
        for name, entry in self.stats.items():
            sampled = np.asarray(entry['values'], dtype=np.float64)
            layers[name] = {k: v for k, v in entry.items() if k != 'values'}
            layers[name].update(scale=max(entry['max_abs'] / 127, 1e-12),
                method='symmetric-minmax-all-observed-elements',
                diagnostic_sample_count=len(sampled),
                diagnostic_p999_abs=float(np.quantile(np.abs(sampled), .999)),
                diagnostic_sample_sha=hashlib.sha256(sampled.tobytes()).hexdigest())
        return dict(schema=SCHEMA, role=ROLE, layers=layers, contexts=self.contexts,
                    calibration_identity=identity(dict(layers=layers, contexts=self.contexts)))


def quantize_weight(array):
    x = np.asarray(array, dtype=np.float32)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite Conv OIHW weights')
    scales = np.maximum(np.abs(x).reshape(x.shape[0], -1).max(1) / 127, 1e-12).astype(np.float32)
    quantized = np.clip(np.rint(x / scales[:, None, None, None]), -127, 127).astype(np.int8)
    return quantized, scales


class ExportBase(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, noise_labels, conditions):
        return self.model(sample, noise_labels=noise_labels, conditional_inputs=[conditions])


class ExportResample(nn.Module):
    """Upstream unit grouped convolutions with statically shaped ONNX weights.

    The original helper constructs its kernel from x.shape[1]. Legacy ONNX
    loses that kernel shape after a dynamic-batch Clip, despite fixed channels.
    Keep the same convolution arithmetic, but make its fixed ones a buffer.
    """
    def __init__(self, channels, mode):
        super().__init__()
        if mode not in ('up', 'down'):
            raise ValueError('Expected nearest unit-convolution up/down resampling')
        self.channels, self.mode = channels, mode
        size = 2 if mode == 'up' else 1
        self.register_buffer('unit_kernel', torch.ones(channels, 1, size, size))

    def forward(self, x):
        operation = F.conv_transpose2d if self.mode == 'up' else F.conv2d
        return operation(x, self.unit_kernel, groups=self.channels, stride=2)


def prepare_export_model(model):
    from terrain_diffusion.models.unet_block import UNetBlock
    converted = copy.deepcopy(model).float().eval()
    for block in converted.modules():
        if isinstance(block, UNetBlock) and block.resample_mode in ('up', 'down'):
            channels = (block.conv_skip.weight.shape[1] * block.conv_skip.groups
                        if block.conv_skip is not None else block.out_channels)
            block.resample = ExportResample(channels, block.resample_mode)
    return converted


def resolve_exclusions(model, exclude=None):
    """Protect the actual input convolution; reject misspelled custom policies."""
    size = getattr(model, 'config', {}).get('image_size')
    if type(size) is not int or size <= 0:
        raise ValueError('Missing valid image_size for input-layer protection')
    first = f'enc.{size}x{size}_conv'
    layers = {name for name, module in model.named_modules() if isinstance(module, FrozenMP)}
    if first not in layers:
        raise ValueError(f'Configured input convolution missing from frozen model: {first}')
    if exclude is None:
        patterns = [p for p in DEFAULT_EXCLUDE if any(p in name for name in layers)]
    else:
        patterns = list(exclude)
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern or not any(pattern in name for name in layers):
                raise ValueError(f'Exclusion matches no frozen layer: {pattern!r}')
    return tuple(dict.fromkeys([first, *patterns]))


def convolution_coverage(node, initializers, calibration_layers, exclude):
    """Account for every Conv/ConvTranspose, including noninitializer weights."""
    if node.op_type not in ('Conv', 'ConvTranspose'):
        return None
    weight = node.input[1] if len(node.input) > 1 else ''
    layer = weight.removeprefix('model.').removesuffix('.weight')
    entry = dict(layer=layer, node_name=node.name, op_type=node.op_type, weight_input=weight)
    if node.op_type == 'ConvTranspose':
        entry['reason'] = 'convtranspose-not-supported-by-this-quantizer'
    elif weight not in initializers:
        entry['reason'] = 'weight-is-not-an-initializer'
    elif any(pattern in layer for pattern in exclude):
        entry['reason'] = 'sensitive-exclusion'
    elif layer not in calibration_layers:
        entry['reason'] = 'no-layer-activation-calibration'
    return entry


def export_qdq(model, example, calibration, directory, *, exclude=None,
               model_kind='base', role=ROLE):
    """Export a float32 conversion plus explicit INT8 Conv Q/DQ. No inference claim."""
    require_scope(model_kind, role)
    if calibration.get('role') != ROLE or not calibration.get('layers'):
        raise ValueError('Missing base preview calibration')
    if any(isinstance(m, MPConv) and not isinstance(m, FrozenMP) for m in model.modules()):
        raise ValueError('Export requires frozen effective MP weights')
    if any(v.device.type != 'cpu' for v in example) or any(p.device.type != 'cpu' for p in (*model.parameters(), *model.buffers())):
        raise ValueError('ONNX export is CPU-only; provide a private frozen CPU copy')
    import onnx
    from onnx import helper, numpy_helper
    exclude = resolve_exclusions(model, exclude)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    converted = prepare_export_model(model)
    plain = directory / 'base-fp32-conversion.onnx'
    with torch.no_grad():
        torch.onnx.export(ExportBase(converted), tuple(x.float() for x in example), str(plain),
            dynamo=False, opset_version=19, input_names=['sample', 'noise_labels', 'conditions'],
            output_names=['prediction'], dynamic_axes={n: {0: 'batch'} for n in
                ('sample', 'noise_labels', 'conditions', 'prediction')}, do_constant_folding=True)
    graph = onnx.load(str(plain))
    initializers = {x.name: x for x in graph.graph.initializer}
    additions, nodes, converted_layers, skipped = [], [], [], []
    for node in graph.graph.node:
        coverage = convolution_coverage(node, initializers, calibration['layers'], exclude)
        if coverage is None:
            nodes.append(node)
            continue
        weight_name = coverage['weight_input']
        layer = coverage['layer']
        if 'reason' in coverage:
            skipped.append(coverage)
            nodes.append(node)
            continue
        weights = numpy_helper.to_array(initializers[weight_name])
        q, scales = quantize_weight(weights)
        prefix = f'terrain_int8_{len(converted_layers)}'
        a_scale = float(calibration['layers'][layer]['scale'])
        if not np.isfinite(a_scale) or a_scale <= 0:
            raise ValueError(f'Invalid activation scale: {layer}')
        for suffix, array in (('w', q), ('ws', scales), ('wz', np.zeros(len(scales), np.int8)),
                              ('as', np.asarray(a_scale, np.float32)), ('az', np.asarray(0, np.int8))):
            additions.append(numpy_helper.from_array(array, prefix+'_'+suffix))
        nodes += [helper.make_node('QuantizeLinear', [node.input[0], prefix+'_as', prefix+'_az'], [prefix+'_aq']),
                  helper.make_node('DequantizeLinear', [prefix+'_aq', prefix+'_as', prefix+'_az'], [prefix+'_adq']),
                  helper.make_node('DequantizeLinear', [prefix+'_w', prefix+'_ws', prefix+'_wz'], [prefix+'_wdq'], axis=0)]
        node.input[0], node.input[1] = prefix+'_adq', prefix+'_wdq'
        nodes.append(node)
        converted_layers.append(dict(**coverage, activation_scale=a_scale, weight_scales_sha=tensor_sha(torch.from_numpy(scales))))
    if not converted_layers:
        raise ValueError('No calibrated eligible convolution exported; refusing empty quantization')
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    graph.graph.initializer.extend(additions)
    used = {name for node in nodes for name in node.input}
    retained = [x for x in graph.graph.initializer if x.name in used]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(retained)
    onnx.checker.check_model(graph)
    target = directory / 'base-int8-qdq.onnx'
    onnx.save(graph, str(target))
    report = dict(schema=SCHEMA, role=ROLE, runtime_enabled=False, state='prepared-not-validated',
        conversion='effective-BF16-weights-promoted-to-FP32; operations-FP32; not-BF16-equivalent',
        quantization='Conv-W8-per-output-channel-A8-static-symmetric-minmax',
        calibration_identity=calibration['calibration_identity'], exclude=list(exclude),
        converted_layers=converted_layers, skipped_layers=skipped,
        files={p.name: sha_file(p) for p in (plain, target)}, runtime=dict(runtime_identity(), onnx=onnx.__version__),
        integer_gpu_execution_proven=False, physical_holdout_passed=False)
    report['artifact_identity'] = identity(report)
    write_json(directory/'export.json', report)
    return report


def tensorrt_network_flags(trt):
    """Prepared API contract for TensorRT10.x; not an engine validation claim.

    NVIDIA10.x precision-control docs require STRONGLY_TYPED and prohibit
    precision flags such as INT8 for these explicitly typed Q/DQ networks.
    Newer majors change API contracts: review before claiming support.
    """
    if str(trt.__version__).split('.')[0] != '10':
        raise ValueError('Optional builder currently targets TensorRT10.x strong typing only; pin a compatible build')
    flag = getattr(trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED', None)
    if flag is None:
        raise ValueError('TensorRT build does not expose STRONGLY_TYPED')
    return 1 << int(flag)


def build_tensorrt(onnx_path, directory, *, workspace_mib=512, max_batch=16, role=ROLE):
    """Optional offline build/inspection. No execution, no claim of an INT8 speedup."""
    require_scope(role=role)
    if not 1 <= max_batch <= 16 or not 1 <= workspace_mib <= 2048:
        raise ValueError('Bounded builder requires batch1..16 and workspace1..2048MiB')
    onnx_path = Path(onnx_path)
    receipt = json.loads((onnx_path.parent/'export.json').read_text(encoding='utf-8'))
    require_scope(role=receipt['role'])
    if receipt.get('artifact_identity') != identity({k:v for k,v in receipt.items() if k != 'artifact_identity'}):
        raise ValueError('Export manifest identity mismatch')
    if (not receipt.get('converted_layers') or receipt.get('runtime_enabled') is not False
            or receipt.get('files', {}).get(onnx_path.name) != sha_file(onnx_path)
            or onnx_path.name != 'base-int8-qdq.onnx'):
        raise ValueError('Builder requires the validated identity of a prepared base Q/DQ export')
    import tensorrt as trt
    network_flags = tensorrt_network_flags(trt)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024**2)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        if shape[0] != -1 or any(d <= 0 for d in shape[1:]):
            raise ValueError(f'Unsupported dynamic input: {tensor.name}/{shape}')
        profile.set_shape(tensor.name, (1, *shape[1:]), (max_batch, *shape[1:]), (max_batch, *shape[1:]))
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT failed to build the explicit Q/DQ network')
    plan_path = directory/'base-preview.plan'
    plan_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('Built engine cannot be deserialized')
    inspector = engine.create_engine_inspector()
    inspection = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    (directory/'layers.json').write_text(inspection, encoding='utf-8')
    report = dict(schema=SCHEMA, role=ROLE, runtime_enabled=False, state='built-not-executed',
        onnx_sha=sha_file(onnx_path), plan_sha=sha_file(plan_path), tensorrt=trt.__version__,
        network_typing='STRONGLY_TYPED', precision_builder_flags=[],
        workspace_limit_mib=workspace_mib, max_batch=max_batch, build_seconds=time.perf_counter()-started,
        engine_device_memory_bytes=int(engine.device_memory_size),
        integer_gpu_execution_proven=False, physical_holdout_passed=False,
        warning='Inspect layers and profile integer kernels; Q/DQ does not prove INT8 coverage or acceleration')
    write_json(directory/'build.json', report)
    return report
