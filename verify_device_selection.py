"""Mock two visible GPUs and verify selection on a newly created worker thread."""
import json
from types import SimpleNamespace
import threading
from unittest.mock import patch

import terrain_device

def main():
    fake = [SimpleNamespace(name='RTX 4090', total_memory=24 * 1024**3,
                            major=8, minor=9, multi_processor_count=128, clock_rate=2520000, uuid='mock4090'),
            SimpleNamespace(name='RTX 5090', total_memory=32 * 1024**3,
                            major=12, minor=0, multi_processor_count=170, clock_rate=2407000, uuid='mock5090')]
    thread_state = threading.local()
    calls = []
    errors = []
    def set_device(index):
        thread_state.device = index
        calls.append((threading.current_thread().name, index))
    def worker():
        try:
            selected = terrain_device.select_cuda_device()
            assert selected['selected']['index'] == 1
            assert thread_state.device == 1
        except Exception as exc:
            errors.append(exc)
    with patch.object(terrain_device.torch.cuda, 'is_available', return_value=True), \
         patch.object(terrain_device.torch.cuda, 'device_count', return_value=2), \
         patch.object(terrain_device.torch.cuda, 'get_device_properties', side_effect=lambda index: fake[index]), \
         patch.object(terrain_device.torch.cuda, 'set_device', side_effect=set_device), \
         patch.object(terrain_device.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='mockdriver\n')), \
         patch.dict(terrain_device.os.environ, {'TERRAIN_CUDA_DEVICE': 'auto'}):
        terrain_device._selection = None
        selected = terrain_device.select_cuda_device()
        assert selected['selected']['index'] == 1 and thread_state.device == 1
        thread = threading.Thread(target=worker, name='mock-gpu-worker')
        thread.start()
        thread.join()
        assert not errors, errors
        assert ('mock-gpu-worker', 1) in calls
        terrain_device._selection = None
        with patch.dict(terrain_device.os.environ, {'TERRAIN_CUDA_DEVICE': '0'}):
            assert terrain_device.select_cuda_device()['selected']['index'] == 0
        terrain_device._selection = None
        with patch.dict(terrain_device.os.environ, {'TERRAIN_CUDA_DEVICE': '2'}):
            try:
                terrain_device.select_cuda_device()
                raise AssertionError('Invalid ordinal accepted')
            except ValueError:
                pass
    terrain_device._selection = None
    print(json.dumps({'passed': True, 'auto_prefers_5090_proxy': True,
                      'worker_ordinal_selected': True, 'explicit_zero': True, 'invalid_rejected': True}))

if __name__ == '__main__':
    main()
