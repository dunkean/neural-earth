"""GPU gate for batch-one learned coarse persistence and cold-store replay."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from terrain_app import RUNTIME, load_pipeline
from terrain_coarse import CoarsePreparation
from terrain_inference import VERSION
from terrain_manifest import build_manifest
from terrain_window_scheduler import read_rect


@torch.inference_mode()
def main():
    world = load_pipeline(42)
    manifest = build_manifest(42, numerical_profile=VERSION,
                              inference_profile=asdict(world._terrain_profile))
    with tempfile.TemporaryDirectory(dir=RUNTIME, prefix='coarse-window-gate-') as temporary:
        world.rebuild()
        prep = CoarsePreparation(temporary, manifest).install(world)
        first_step = prep.step(world, budget_windows=1)
        before = read_rect(world, 'coarse', -2, -2, 2, 2).cpu().numpy()
        prep.flush()
        first = prep.status()
        world.rebuild()
        replay_prep = CoarsePreparation(temporary, manifest).install(world)
        after = read_rect(world, 'coarse', -2, -2, 2, 2).cpu().numpy()
        replay_prep.flush()
        finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
        error = float(np.max(np.abs(before - after)))
        positive_denominator = bool(np.all(after[-1] > 0))
        sources = {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                   for name in ('terrain_coarse.py', 'terrain_window_scheduler.py',
                                'terrain_inference.py', 'verify_coarse_persistence.py')}
        report = {'world_hash': manifest['world_hash'], 'first_step': first_step,
                  'after_generation': first, 'after_replay': replay_prep.status(),
                  'max_weighted_channel_error': error,
                  'height_denominator_positive': positive_denominator,
                  'replay_network_windows': replay_prep.network_windows,
                  'replay_disk_hits': replay_prep.disk_hits,
                  'finite': finite, 'gpu': torch.cuda.get_device_name(),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'source_sha256': sources,
                  'passed': bool(finite and error == 0 and positive_denominator and
                                 replay_prep.network_windows == 0 and
                                 replay_prep.disk_hits > 0 and
                                 first_step['network_windows'] == 1)}
        replay_prep.close()
        prep.close()
    path = RUNTIME / 'coarse-persistence-fidelity.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise AssertionError('Learned coarse persistence gate failed')


if __name__ == '__main__':
    main()
