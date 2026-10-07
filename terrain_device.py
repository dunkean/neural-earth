"""Select one visible CUDA device before model/profile/cache initialisation."""
import os
import subprocess
import torch

_selection = None


def select_cuda_device():
    global _selection
    if _selection is not None:
        # CUDA's current ordinal is thread-local. Each new GPU-worker/request
        # thread must select the recorded device too.
        torch.cuda.set_device(_selection['selected']['index'])
        return _selection
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: the local Terrain Diffusion agent requires a compatible NVIDIA GPU.')
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
