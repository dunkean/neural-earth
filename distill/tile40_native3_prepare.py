"""Capture native teacher windows with explicit paired climates,TRAIN/VAL only."""
import argparse
import gc
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, HOLDOUT_SEEDS, atomic_json, external_path
from distill.native_base import CaptureCalls
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.teacher import Worlds
from distill.tile40_conditioning import CLIMATES, RELIEFS, SketchConditioning


def selected_cases(manifest, mountains=12, other=4):
    result = []
    for split in ('train', 'val'):
        for relief in RELIEFS:
            rows = [c for c in manifest['cases'] if c['split']==split and c['relief']==relief]
            seeds = sorted({c['seed'] for c in rows})[:(mountains if relief=='mountains' else other) if split=='train' else 1]
            if any(seed in HOLDOUT_SEEDS for seed in seeds):
                raise ValueError('Reserved world entered the capture bank.')
            for seed in seeds:
                for climate in CLIMATES:
                    result.append(next(c for c in rows if c['seed']==seed and c['climate']==climate))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-coarse2-guided-expanded-v5')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mountains', type=int, default=12)
    parser.add_argument('--other', type=int, default=4)
    parser.add_argument('--per-stage', type=int, default=16)
    args = parser.parse_args()
    if min(args.mountains, args.other, args.per_stage)<1:
        raise ValueError('Positive capture quotas required.')
    root = external_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = args.dataset/'manifest.json'
    source = json.loads(source_manifest.read_text())
    cases = selected_cases(source, args.mountains, args.other)
    contract = dict(schema=1, teacher_revision=REVISION, source_dataset=str(args.dataset),
        source_manifest_sha256=file_sha(source_manifest), producer_sha256=file_sha(Path(__file__)),
        coarse_sha256=file_sha(REPO/'distill/weights/coarse2.pt'), cases=cases,
        per_stage=args.per_stage, latent_bounds=[48,48,112,112],
        purpose='Original native64px two-stage teacher calls;paired climates on same TRAIN geometry.',
        split_policy='No TEST;independent TRAIN/VAL geometry;captured targets never mixed across splits.')
    path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Capture contract changed;preserve the previous bank.')
    atomic_json(path, contract)
    stopped = False
    def stop(signum,frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    torch.set_num_threads(6)
    worlds = None
    coarse = None
    began = time.monotonic()
    for index,case in enumerate(cases):
        destination = root/case['file']
        if destination.exists():
            with np.load(destination,allow_pickle=False) as saved:
                meta = json.loads(str(saved['metadata']))
                if meta['case']!=case or meta['capture_manifest_sha256']!=file_sha(path):
                    raise ValueError('Captured data differs from contract.')
            continue
        if stopped:
            break
        source_path = args.dataset/case['file']
        if file_sha(source_path)!=json.loads(source_path.with_suffix('.json').read_text())['file_sha256']:
            raise ValueError('Frozen source changed.')
        with np.load(source_path,allow_pickle=False) as saved:
            elevation = saved['sketch_m'].copy()
        if worlds is None:
            worlds = Worlds(case['seed'])
            coarse,_ = load_student(REPO/'distill/weights/coarse2.pt','cuda')
        world = worlds.create(case['seed'],'natural')
        from distill.inference import install
        install(world,dict(coarse=coarse))
        world.bind()
        world.synthetic_map_factory = SketchConditioning(elevation,case['climate'])
        metadata = dict(case=case,split=case['split'],seed=case['seed'],climate=case['climate'],
            relief=case['relief'],capture_manifest_sha256=file_sha(path),
            source_file_sha256=file_sha(source_path),teacher_revision=REVISION)
        recorder = CaptureCalls(world.base_model,destination,metadata,limit=args.per_stage)
        world.base_model = recorder
        try:
            world.latents[:,48:112,48:112]
            count = recorder.save()
            atomic_json(destination.with_suffix('.json'),dict(metadata=metadata,
                file_sha256=file_sha(destination),windows=count))
            print(json.dumps(dict(case=case,windows=count,completed=index+1,total=len(cases))),flush=True)
            atomic_json(root/'status.json',dict(status='running',completed=index+1,total=len(cases),elapsed_seconds=time.monotonic()-began))
        finally:
            world.close()
            del world,recorder
            gc.collect()
            torch.cuda.empty_cache()
    available = sum((root/c['file']).exists() for c in cases)
    atomic_json(root/'status.json',dict(status='complete' if available==len(cases) else 'interrupted',
        completed=available,total=len(cases),elapsed_seconds=time.monotonic()-began))


if __name__=='__main__':
    main()
