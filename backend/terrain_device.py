"""Visible CUDA devices, the multi-GPU plan and per-thread device binding.

``select_cuda_device`` keeps its historical contract: one *primary* device is
chosen before model/profile/cache initialisation. The multi-GPU plan then
assigns roles on top of it:

* the **coarse** device computes every learned coarse window. Its 20-step BF16
  solver amplifies hardware differences (tens of metres between a 4090 and a
  5090), so a world never mixes coarse windows from two GPU models;
* **detail** devices run latent/decoder tiles. Different GPU models differ by
  a few metres at most there, which only matters between neighbouring tiles.

Modes: ``single`` (primary only), ``throughput`` (coarse on the primary,
detail tiles on every enabled device), ``consistent`` (coarse on one device,
detail tiles on another; each window type always comes from the same GPU) and
``auto`` (throughput when at least two eligible GPUs are visible).
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid as _uuid
import torch

GPU_MODES = ('auto', 'single', 'throughput', 'consistent')
# BF16 tensor cores and enough memory for a second full model copy and caches.
MIN_MULTI_GPU_COMPUTE = (8, 0)
MIN_MULTI_GPU_MEMORY = 10 * 1024**3
SETTINGS_SCHEMA = 'neural-earth-gpu-settings-v1'

_selection = None
_thread = threading.local()


def select_cuda_device():
    global _selection
    if _selection is not None:
        # CUDA's current ordinal is thread-local. Each new GPU-worker/request
        # thread must select the recorded device too; a worker bound to another
        # plan device keeps its own ordinal.
        torch.cuda.set_device(current_cuda_index())
        return _selection
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: Neural Earth requires a compatible NVIDIA GPU.')
    requested = os.environ.get('TERRAIN_CUDA_DEVICE', 'auto').strip().lower()
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        # Hardware proxy only: SM count * advertised CUDA clock. This is useful
        # for 3090/4090/5090 selection, but not a measured inference ranking.
        score = int(props.multi_processor_count) * int(props.clock_rate)
        devices.append({'index': index, 'name': props.name, 'uuid': str(props.uuid),
                        'memory_bytes': props.total_memory, 'compute_capability': f'{props.major}.{props.minor}',
                        'sm_count': props.multi_processor_count, 'clock_khz': props.clock_rate,
                        'hardware_score': score})
    if requested == 'auto':
        chosen = max(devices, key=lambda device: (device['hardware_score'], device['memory_bytes'], device['compute_capability']))
    else:
        try:
            index = int(requested)
        except ValueError as exc:
            raise ValueError('TERRAIN_CUDA_DEVICE must be auto or a visible CUDA index (0, 1, ...)') from exc
        if not 0 <= index < len(devices):
            raise ValueError(f'TERRAIN_CUDA_DEVICE={index}: only {len(devices)} visible CUDA device(s)')
        chosen = devices[index]
    torch.cuda.set_device(chosen['index'])
    driver = None
    try:
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=3, check=False, **kwargs)
        if result.returncode == 0:
            driver = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        pass
    _selection = {'requested': requested, 'selected': chosen, 'visible_devices': devices,
                  'selection_method': 'explicit CUDA ordinal' if requested != 'auto' else 'unbenchmarked SM-count × clock hardware heuristic',
                  'driver': driver, 'multi_gpu_workers': False}
    return _selection


def current_cuda_index():
    """The ordinal bound to this thread, otherwise the primary selection."""
    index = getattr(_thread, 'index', None)
    if index is not None:
        return index
    return _selection['selected']['index'] if _selection else 0


def bind_thread_device(index):
    """Pin this thread (a GPU worker) to one plan device for its lifetime."""
    _thread.index = index
    if torch.cuda.is_available():
        torch.cuda.set_device(index)


@contextmanager
def using_device(index):
    """Temporarily run this thread on another plan device, then restore it."""
    previous = getattr(_thread, 'index', None)
    bind_thread_device(index)
    try:
        yield
    finally:
        _thread.index = previous
        if torch.cuda.is_available():
            torch.cuda.set_device(current_cuda_index())


def set_multi_gpu_workers(enabled):
    if _selection is not None:
        _selection['multi_gpu_workers'] = bool(enabled)


def multi_gpu_eligible(device):
    major, minor = (int(part) for part in str(device['compute_capability']).split('.')[:2])
    return (major, minor) >= MIN_MULTI_GPU_COMPUTE and device['memory_bytes'] >= MIN_MULTI_GPU_MEMORY


def _rank(device):
    return (device['hardware_score'], device['memory_bytes'], device['compute_capability'])


def _lookup(devices, value):
    """Resolve a persisted UUID or a CUDA ordinal (environment) to a device."""
    for device in devices:
        if value == device['uuid'] or (isinstance(value, int) and not isinstance(value, bool)
                                       and value == device['index']):
            return device
    return None


def default_gpu_settings():
    return {'mode': 'auto', 'devices': 'auto', 'coarse_device': 'auto'}


def validate_gpu_settings(settings, devices):
    """Normalise a client/persisted settings object; raise ValueError if invalid."""
    if not isinstance(settings, dict):
        raise ValueError('GPU settings must be an object')
    unknown = set(settings) - {'mode', 'devices', 'coarse_device'}
    if unknown:
        raise ValueError(f'Unknown GPU settings: {sorted(unknown)}')
    result = default_gpu_settings()
    mode = settings.get('mode', 'auto')
    if mode not in GPU_MODES:
        raise ValueError(f'GPU mode must be one of {", ".join(GPU_MODES)}')
    result['mode'] = mode
    selected = settings.get('devices', 'auto')
    if selected != 'auto':
        if not isinstance(selected, list) or not selected or len(selected) > 64:
            raise ValueError('GPU devices must be "auto" or a non-empty list')
        resolved = []
        for value in selected:
            device = _lookup(devices, value)
            if device is None:
                raise ValueError(f'Unknown GPU: {value}')
            if device['uuid'] not in resolved:
                resolved.append(device['uuid'])
        selected = resolved
    result['devices'] = selected
    coarse = settings.get('coarse_device', 'auto')
    if coarse != 'auto':
        device = _lookup(devices, coarse)
        if device is None:
            raise ValueError(f'Unknown coarse GPU: {coarse}')
        coarse = device['uuid']
    result['coarse_device'] = coarse
    return result


def environment_gpu_settings(environ=None):
    """Startup overrides. They take precedence over settings saved from the UI."""
    environ = os.environ if environ is None else environ
    overrides = {}
    if environ.get('TERRAIN_GPU_MODE'):
        overrides['mode'] = environ['TERRAIN_GPU_MODE'].strip().lower()
    elif environ.get('TERRAIN_CUDA_DEVICE', 'auto').strip().lower() != 'auto':
        # An explicitly pinned ordinal keeps its historical single-GPU meaning.
        overrides['mode'] = 'single'
    if environ.get('TERRAIN_CUDA_DEVICES'):
        try:
            overrides['devices'] = [int(part) for part in environ['TERRAIN_CUDA_DEVICES'].split(',') if part.strip()]
        except ValueError as exc:
            raise ValueError('TERRAIN_CUDA_DEVICES must list CUDA ordinals, e.g. 0,1') from exc
    if environ.get('TERRAIN_COARSE_DEVICE', 'auto').strip().lower() != 'auto':
        try:
            overrides['coarse_device'] = int(environ['TERRAIN_COARSE_DEVICE'])
        except ValueError as exc:
            raise ValueError('TERRAIN_COARSE_DEVICE must be auto or a CUDA ordinal') from exc
    return overrides


def resolve_gpu_plan(devices, primary, settings=None):
    """Assign coarse/detail roles. Pure function of the inventory and settings."""
    settings = validate_gpu_settings(settings or {}, devices)
    by_index = {device['index']: device for device in devices}
    if primary not in by_index:
        raise ValueError('Primary GPU is not visible')
    enabled = ([device for device in devices if device['uuid'] in settings['devices']]
               if settings['devices'] != 'auto' else list(devices))
    # The primary is always usable: it is what the single-GPU server would use.
    candidates = [by_index[primary]] + sorted(
        (device for device in enabled if device['index'] != primary and multi_gpu_eligible(device)),
        key=_rank, reverse=True)
    coarse_override = _lookup(devices, settings['coarse_device']) if settings['coarse_device'] != 'auto' else None
    if coarse_override is not None and coarse_override not in candidates:
        candidates.append(coarse_override)
    requested = settings['mode']
    mode = requested
    reason = None
    if mode == 'auto':
        mode = 'throughput' if len(candidates) >= 2 else 'single'
        reason = ('two or more eligible GPUs' if mode == 'throughput'
                  else 'only one eligible GPU (compute capability >= 8.0, >= 10 GiB)')
    if mode in ('throughput', 'consistent') and len(candidates) < 2:
        reason = f'{mode} needs two eligible GPUs; using one'
        mode = 'single'
    if mode == 'single':
        coarse = coarse_override or candidates[0]
        detail = [coarse]
    elif mode == 'throughput':
        coarse = coarse_override or candidates[0]
        detail = [coarse] + [device for device in candidates if device is not coarse]
    else:
        coarse = coarse_override or candidates[1]
        detail = [next(device for device in candidates if device is not coarse)]
    used = []
    for device in [coarse, *detail]:
        if device['index'] not in used:
            used.append(device['index'])
    return {'requested_mode': requested, 'mode': mode, 'reason': reason,
            'coarse_device': coarse['index'], 'detail_devices': [device['index'] for device in detail],
            'devices': used, 'settings': settings,
            'eligible': [device['index'] for device in devices if multi_gpu_eligible(device)]}


def load_gpu_settings(path, devices):
    """Saved UI choice; unreadable or stale files fall back to defaults."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema') != SETTINGS_SCHEMA:
            return default_gpu_settings()
        return validate_gpu_settings({k: data[k] for k in ('mode', 'devices', 'coarse_device') if k in data}, devices)
    except (OSError, ValueError, TypeError, AttributeError):
        return default_gpu_settings()


def save_gpu_settings(path, settings):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{_uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(dict(settings, schema=SETTINGS_SCHEMA), indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
