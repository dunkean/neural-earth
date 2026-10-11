"""Diverse teacher windows without generating unused decoder data in training."""
import json

import numpy as np
import torch

from distill.native_mixed_bank import KINDS, VARIANTS
from distill.native_phase_loss import PHASE_LABELS


def choose_windows(records, generator, step):
    """Eight worlds, four windows per phase, two rare cases, all coarse branches."""
    selected, used = [], set()
    for phase in (0, 1):
        variants = list(VARIANTS)+[VARIANTS[(step+phase) % len(VARIANTS)]]
        generator.shuffle(variants)
        rare_slot = int(generator.integers(4))
        kind = KINDS[(step+2*phase) % len(KINDS)]
        for slot, variant in enumerate(variants):
            case = kind if slot == rare_slot else None
            pool = [r for r in records if r['coarse_variant'] == variant and r['case_kind'] == case
                    and r['metadata']['seed'] not in used]
            profiles = sorted({r['metadata']['profile'] for r in pool})
            if not profiles:
                raise ValueError('Insufficient independent TRAIN worlds in the selected stratum.')
            profile = profiles[int(generator.integers(len(profiles)))]
            pool = [r for r in pool if r['metadata']['profile'] == profile]
            record = pool[int(generator.integers(len(pool)))]
            used.add(record['metadata']['seed']); selected.append((phase, record))
    return selected


def read_window(path, phase, generator, device):
    with np.load(path, allow_pickle=False) as sample:
        metadata = json.loads(str(sample['metadata']))
        if metadata['native_size'] != 128:
            raise ValueError('Native128 teacher scene required.')
        core = tuple(((metadata[key]+95)//96)*96 for key in ('y', 'x'))
        offsets = (-64, -32, 0, 32, 64) if phase == 0 else (-32, 0, 32)
        origins = {(core[0]+dy, core[1]+dx) for dy in offsets for dx in offsets}
        labels, contexts = sample['labels'], sample['contexts']
        ids = [int(i) for i in np.flatnonzero(labels == PHASE_LABELS[phase])
               if (int(contexts[i, 1])*32, int(contexts[i, 2])*32) in origins]
        if len(ids) != len(origins):
            raise ValueError('Complete teacher windows in the chosen phase required.')
        index = ids[int(generator.integers(len(ids)))]
        return {key: torch.from_numpy(sample[key][index:index+1].copy()).to(device)
                for key in ('inputs', 'labels', 'condition', 'target')}


def diverse_batch(records, root, generator, step, device='cuda'):
    selected = choose_windows(records, generator, step)
    windows = [read_window(root/r['file'], phase, generator, device) for phase, r in selected]
    return {key: torch.cat([w[key] for w in windows]) for key in windows[0]}, selected
