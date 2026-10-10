"""Export a preserved candidate bundle without optimizer or duplicate EMA weights.

Exports keep the exact EMA used for evaluation. Their manifest records training
sources and hashes; exporting does not assert quality or authorize deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from distill.common import REVISION, atomic_json, atomic_write, external_path


def export_bundle(checkpoints, output):
    if set(checkpoints) != {'base', 'coarse', 'decoder'}:
        raise ValueError('A bundle requires all three stages.')
    sources, saved = {}, {}
    for stage, path in checkpoints.items():
        payload = Path(path).read_bytes()
        source = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if source['config']['stage'] != stage or source['step'] <= 0:
            raise ValueError(f'{stage}: a trained checkpoint of the matching stage is required.')
        state = source.get('ema', source['model'])
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f'{stage}: non-finite weights.')
        sources[stage] = dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                              step=source['step'], config=source['config'])
        saved[stage] = dict(config=source['config'], model=state, step=source['step'],
                            inference_only=True, selected_weights='ema' if 'ema' in source else 'model',
                            source_checkpoint=sources[stage], teacher_revision=REVISION,
                            seed_plan=source.get('seed_plan'), warm_start=source.get('warm_start'))
    output = external_path(output)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # JSON converts configuration tuples to lists.
        if manifest['sources'] != json.loads(json.dumps(sources)):
            raise ValueError('An exported bundle is immutable; use a separately named output.')
        for stage, artifact in manifest['exports'].items():
            if hashlib.sha256((output/artifact['file']).read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError(f'{stage}: exported weights changed.')
        return manifest
    artifacts = {}
    for stage, value in saved.items():
        path = output/(stage+'.pt')
        atomic_write(path, lambda handle: torch.save(value, handle))
        artifacts[stage] = dict(file=path.name, bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                tensor_elements=sum(t.numel() for t in value['model'].values()))
    manifest = dict(schema=1, sources=sources, exports=artifacts, inference_only=True,
                    teacher_model='xandergos/terrain-diffusion-30m', teacher_revision=REVISION,
                    accepted=False, note='Preserved EMA candidates, with no optimizer state. See physical, seam and speed evidence before selection.')
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ('base', 'coarse', 'decoder'):
        parser.add_argument('--'+stage, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = export_bundle({stage:getattr(args, stage) for stage in ('base', 'coarse', 'decoder')}, args.output)
    print(json.dumps(manifest['exports']), flush=True)


if __name__ == '__main__':
    main()
