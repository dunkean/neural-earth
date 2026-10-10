"""Recover rejected synthetic worlds without changing existing teacher targets.

Keeps teacher.py and its source digest unchanged. Replacement worlds use the
same original teacher implementation with a recorded deterministic new seed.
The sidecar is part of dataset/checkpoint provenance and audited separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from distill.common import DATA, HOLDOUT_SEEDS, atomic_json

STRIDE = 1000000


def load_plan(root, manifest):
    path = Path(root)/'seed-replacements.json'
    if not path.exists():
        return {}
    plan = json.loads(path.read_text())
    if plan['policy']['teacher_sha256'] != manifest['source_digest']['distill/teacher.py'] or plan['policy']['stride'] != STRIDE:
        raise ValueError('Seed replacement policy does not match the teacher provenance.')
    used = set()
    for original, record in plan['replacements'].items():
        world = int(original)-manifest['seed_base']
        seed = record['seed']
        if (world < 0 or not 1 <= record['attempt'] <= 8 or seed != int(original)+STRIDE*record['attempt'] or
                seed in HOLDOUT_SEEDS or seed in used or
                record['profile'] != manifest['profiles'][world % len(manifest['profiles'])]):
            raise ValueError('Invalid/colliding seed replacement.')
        used.add(seed)
    return plan


def validate_resume_plan(saved, current, files, manifest, allow_growth):
    if saved == current:
        return
    if not allow_growth:
        raise ValueError('Seed replacements changed; explicit data growth is required.')
    if saved and saved['policy'] != current.get('policy'):
        raise ValueError('Seed replacement policy changed.')
    before, after = saved.get('replacements', {}), current.get('replacements', {})
    if any(after.get(key) != value for key, value in before.items()):
        raise ValueError('An existing seed replacement changed or disappeared.')
    old_worlds = {int(Path(name).stem.split('-')[-1])//manifest['samples_per_world'] for name in files}
    if any(int(key)-manifest['seed_base'] in old_worlds for key in after.keys()-before.keys()):
        raise ValueError('Seed replacement would change an existing training example.')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, default=DATA/'crops/main')
    args, _ = parser.parse_known_args()
    root = args.output.expanduser().resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    path = root/'seed-replacements.json'
    policy = dict(stride=STRIDE, teacher_sha256=manifest['source_digest']['distill/teacher.py'],
                  recovery_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan = load_plan(root, manifest) or dict(policy=policy, replacements={})
    if plan['policy'] != policy:
        raise ValueError('Recovery implementation changed; inspect provenance before resuming.')
    from distill import teacher
    create, generate = teacher.Worlds.create, teacher.generate_sample

    def create_world(self, seed, profile):
        key = str(seed)
        if key in plan['replacements']:
            return create(self, plan['replacements'][key]['seed'], profile)
        try:
            return create(self, seed, profile)
        except RuntimeError as exc:
            if 'topology did not pass in 12 deterministic candidates' not in str(exc):
                raise
            reason = str(exc)[:1200]
        world_index = seed-manifest['seed_base']
        first = world_index*manifest['samples_per_world']
        # Never assign a different world to any already committed example,
        # including held-out examples generated before the training prefix.
        for stage in manifest['stages']:
            for index in range(first, first+manifest['samples_per_world']):
                for split in ('train', 'val'):
                    if (root/stage/f'{split}-{index:07d}.npz').exists():
                        raise ValueError('Rejected world already has targets; refuse to change their seed.')
        for attempt in range(1, 9):
            replacement = seed+STRIDE*attempt
            if replacement in HOLDOUT_SEEDS or any(r['seed'] == replacement for r in plan['replacements'].values()):
                continue
            try:
                world = create(self, replacement, profile)
            except RuntimeError as exc:
                if 'topology did not pass in 12 deterministic candidates' not in str(exc):
                    raise
                continue
            record = dict(seed=replacement, profile=profile, attempt=attempt, reason=reason,
                          recorded_at=time.time())
            plan['replacements'][key] = record
            atomic_json(path, plan)  # durable before the first new target
            print(json.dumps(dict(replaced_seed=seed, **record)), flush=True)
            return world
        raise RuntimeError(f'No valid replacement for seed {seed}, profile {profile}.')

    def generate_actual_seed(world, index, seed, profile, split, output, stages, crop, halo):
        return generate(world, index, world.seed, profile, split, output, stages, crop, halo)

    teacher.Worlds.create = create_world
    teacher.generate_sample = generate_actual_seed
    try:
        teacher.main()
    except Exception as exc:
        # Original orchestration can mask a failed create() with an unbound
        # cleanup variable. Preserve the original cause in the durable status.
        cause = exc.__context__ if isinstance(exc, UnboundLocalError) and exc.__context__ else exc
        atomic_json(root/'status.json', dict(status='failed', error=str(cause), updated_at=time.time()))
        if cause is not exc:
            raise cause from exc
        raise


if __name__ == '__main__':
    main()
