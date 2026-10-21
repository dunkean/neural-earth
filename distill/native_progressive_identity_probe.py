"""Bounded exact-source checks of old KD and new physical recovery losses."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_auxiliary_training import BoundedQATBank, gradient_norm
from distill.native_base import teacher_source
from distill.native_basis_training import inventory, capture_blocks
from distill.native_level_objective import full_level_fields, level_terms
from distill.native_progressive_prepare import configure, feature_loss
from distill.native_progressive_recovery_loss import x0_height_terms, field_terms
from distill.native_progressive_width import ProgressiveWidthBase
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_train import loss_terms
from terrain_diffusion.models.edm_unet import EDMUnet2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the previous identity proof.')
    configure()
    source_path = args.preparation/'initial.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    proof = json.loads((args.preparation/'proof.json').read_text())
    if source['amount'] != 0. or proof['initial_sha256'] != file_sha(source_path):
        raise ValueError('Verified zero-removal initial weights are required.')
    model = ProgressiveWidthBase(source['config'], source['target_config'], source['mapping'], source['gain_anchor']).cuda()
    model.load_state_dict(source['ema'])
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-5, betas=(.9, .99), fused=True)
    records, _, _ = inventory()
    bank = BoundedQATBank(limit=32)
    variance = torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'], device='cuda')
    floors = json.loads((DATA/'eval/native-nf4-level-train-floor-4090.json').read_text())['floors']
    batches = []
    for step in range(4):
        with torch.no_grad(), model.cached_weights():
            batch, _ = bank.batch(records, np.random.default_rng(2026101361+step), step, model, on_policy=step%2 == 0)
        with torch.no_grad(), capture_blocks(teacher) as reference:
            desired = teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                              conditional_inputs=[batch['condition'].bfloat16()]).float()
        with capture_blocks(model.net) as actual:
            prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        if not torch.equal(prediction, desired):
            raise ValueError('Exact-source operator changed.')
        main_loss, _ = loss_terms(prediction, desired, batch, variance)
        features = feature_loss(actual, reference, source['mapping'])
        physical = x0_height_terms(prediction, desired, batch)['height_jacobian_mse']
        optimizer.zero_grad(set_to_none=True)
        total = main_loss+proof['feature_weight']*features+physical
        total.backward()
        norm = gradient_norm([p.grad for p in params])
        if float(total.detach()) != 0. or norm != 0.:
            raise ValueError('Matched source receives an unintended regular gradient.')
        optimizer.step()
        batches.append(dict(step=step, on_policy=batch['on_policy'], source_exact=True,
                            loss=0., grad_norm=norm))
        del actual, reference, prediction, desired, total, main_loss, features, physical
    unchanged = all(torch.equal(value, source['ema'][name].to(value.device, dtype=value.dtype))
                    for name, value in model.state_dict().items())
    if not unchanged:
        raise ValueError('Zero-gradient Adam updates changed the source parameters.')
    fields = []
    for index, kind in enumerate(('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', 'wet-high-mountain')):
        record = next(r for r in records if r['case_kind'] == kind and r['coarse_variant'] == 'coarse4')
        optimizer.zero_grad(set_to_none=True)
        actual, desired = full_level_fields(model, teacher, record, 2026101361+index*7919)
        if not torch.equal(actual, desired):
            raise ValueError('Exact-source full two-stage TRAIN field changed: '+kind)
        old = level_terms(actual, desired, variance, floors[kind])['level']
        new = field_terms(actual, desired, floors[kind]['height_lf_rms_m'])
        total = 2e-5*old+new['height_exact_mse']+new['coast']
        total.backward()
        norm = gradient_norm([p.grad for p in params])
        if float(total.detach()) != 0. or norm > 1e-6:
            raise ValueError('Matched two-stage field receives an unintended physical gradient: '+kind)
        fields.append(dict(kind=kind, seed=record['metadata']['seed'], source_exact=True,
                           old_level_loss=float(old.detach()), new_height_loss=float(new['height_exact_mse'].detach()),
                           coast_loss=float(new['coast'].detach()), grad_norm=norm))
        del actual, desired, old, new, total
    if any(p.grad is not None for p in teacher.parameters()):
        raise ValueError('Frozen teacher received a gradient.')
    atomic_json(output, dict(status='complete', source_sha256=file_sha(source_path), batches=batches,
        four_zero_gradient_adam_steps_unchanged=unchanged, fields=fields,
        teacher_frozen=True, gpu=torch.cuda.get_device_name(), peak_bytes=torch.cuda.max_memory_allocated(),
        source_code_sha256=file_sha(REPO/'distill/native_progressive_identity_probe.py'),
        note='Bounded TRAIN identity/gradient proof; not a2k training experiment or a claim that a compressed pupil is converged.'))
    print(json.dumps(dict(status='complete', fields_exact=len(fields), optimizer_unchanged=unchanged)), flush=True)


if __name__ == '__main__':
    main()
