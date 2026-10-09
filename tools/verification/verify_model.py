"""Check real GPU inference, seed switching, climate and reproducibility."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import json
import numpy as np
import torch
from terrain_app import load_pipeline, OUTPUT, gpu_calls

with torch.inference_mode():
    world = load_pipeline(42)
    def sample():
        result = world.get(2560, -3584, 2816, -3328, with_climate=True)
        elevation = result['elev'].float().cpu().numpy()
        climate = result['climate'].float().cpu().numpy()
        assert elevation.shape == (256, 256)
        assert climate.shape == (5, 256, 256)
        assert np.isfinite(elevation).all() and np.isfinite(climate).all()
        return elevation
    first = sample()
    world.change_seed(43)
    other = sample()
    assert not np.allclose(first, other), 'Seed does not change the world'
    world.change_seed(42)
    repeated = sample()
    assert np.array_equal(first, repeated), 'Seed reset must reproduce the same crop'
    assert all(count > 0 for count in gpu_calls.values())
    result = {'passed': True, 'cuda_forward_calls': dict(gpu_calls),
              'seed_changes_terrain': True, 'seed_reset_exact': True,
              'elevation_finite': True, 'climate_shape': [5, 256, 256],
              'climate_finite': True}
    (OUTPUT / 'model-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
