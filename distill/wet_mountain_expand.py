"""Mine additional independent mountain TRAIN worlds from original teacher data."""
import argparse
import json

import numpy as np

from distill.common import DATA, REVISION, atomic_json, external_path, source_digest
from distill.native_base import teacher_source
from distill.rare_cases import field_stats
from distill.wet_mountains import KIND, CRITERIA, ROOT, check_splits, digest, mountain_stats, qualifies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(DATA/'crops/native-wet-mountain-train-extension-v1'))
    parser.add_argument('--worlds', type=int, default=12)
    args = parser.parse_args()
    root = external_path(args.output)
    if (root/'proposals.json').exists():
        raise ValueError('Keep existing proposals immutable.')
    main = DATA/'crops/main'
    coverage = json.loads((DATA/'eval/rare-training-coverage.json').read_text())
    existing = json.loads((DATA/'crops/native-mixed-scenes-v2/manifest.json').read_text())['scenes']
    parent = json.loads((ROOT/'manifest.json').read_text())['scenes']
    excluded = {r['metadata']['seed'] for r in parent}
    candidates = [r for r in coverage['rows'] if r['split']=='train' and r['seed'] not in excluded
                  and r['climate'] and r['climate']['rain']>=1200.
                  and r['teacher_lowfreq']['land_height_p90']>=2000.
                  and r['teacher_lowfreq']['land_height_std']>=300.]
    selected, used = [], set()
    for row in sorted(candidates, key=lambda r:-r['teacher_lowfreq']['land_height_std']):
        if row['seed'] in used: continue
        pool = []
        with np.load(main/'base'/row['file'], allow_pickle=False) as sample:
            meta = json.loads(str(sample['metadata']))
            target, coarse = sample['target'], sample['coarse']
            for y in range(((meta['y']+95)//96)*96, meta['y']+meta['crop']-63, 96):
                for x in range(((meta['x']+95)//96)*96, meta['x']+meta['crop']-63, 96):
                    oy, ox = y-meta['y'], x-meta['x']
                    q = target[4, oy:oy+64, ox:ox+64].astype(np.float64)*38.6-31.4
                    height = np.sign(q)*q**2
                    cy, cx = y//32-meta['coarse_y'], x//32-meta['coarse_x']
                    climate = coarse[2:, cy:cy+2, cx:cx+2].astype(np.float64).repeat(32,1).repeat(32,2)
                    stats, c = mountain_stats(height, 3), field_stats(np.concatenate((height[None],climate)))
                    if qualifies(stats, c):
                        pool.append(dict(metadata=dict(seed=meta['seed'], split='train', profile=meta['profile'],
                            y=y, x=x, teacher_revision=REVISION), case_kind=KIND, proposal_climate=c,
                            proposal_relief=stats, source=str(main/'base'/row['file'])))
        if pool:
            item = max(pool, key=lambda r:r['proposal_relief']['relief_p90_p10_m'])
            selected.append(item); used.add(row['seed'])
        if len(used)>=args.worlds: break
    if len(used)<args.worlds:
        raise ValueError(f'Only {len(used)} qualifying new mountain TRAIN worlds.')
    check_splits(selected, existing+parent)
    plan = dict(schema=1, revision=REVISION, scenes=selected, criteria=CRITERIA,
        teacher_code=source_digest(), teacher_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        parent_manifest_sha256=digest(ROOT/'manifest.json'), coverage_sha256=digest(DATA/'eval/rare-training-coverage.json'),
        frozen=False, accepted=False, note='Additional TRAIN-only original teacher proposals; decoded qualification pending.')
    atomic_json(root/'proposals.json', plan)
    print(json.dumps(dict(worlds=len(used), scenes=len(selected))), flush=True)


if __name__ == '__main__': main()
