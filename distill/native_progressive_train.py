"""First20k stage: slow channel removal192->160, three blocks and two passes."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_auxiliary_training import BoundedQATBank, preconditioners, gradient_norm, combine_gradients
from distill.native_base import teacher_source
from distill.native_basis_training import inventory, capture_blocks
from distill.native_level_objective import full_level_fields, level_terms, protect_scalar_gradients
from distill.native_progressive_prepare import configure, feature_loss
from distill.native_progressive_width import ProgressiveWidthBase
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_train import loss_terms
from terrain_diffusion.models.edm_unet import EDMUnet2D

CODE = ('native_progressive_width.py', 'native_progressive_prepare.py', 'native_progressive_train.py',
        'native_compression.py', 'native_weight_quant_train.py', 'native_auxiliary_training.py',
        'native_level_objective.py', 'native_decoded.py', 'native_fusion.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = external_path(args.output)
    configure()
    proof = json.loads((args.preparation/'proof.json').read_text())
    source = torch.load(args.preparation/'initial.pt', map_location='cpu', weights_only=True, mmap=True)
    if (proof['status'] != 'complete' or proof['initial_sha256'] != file_sha(args.preparation/'initial.pt') or
            any(proof['source_code_sha256'][name] != file_sha(REPO/'distill'/name)
                for name in proof['source_code_sha256'])):
        raise ValueError('Matching exact-source/backward/compact preparation required.')
    contract = dict(steps=args.steps, width=160, blocks=3, passes=2, start_width=192,
        amount='smoothstep from0 to1 over20k; explicit MP, encoder, embedding and concat projections',
        source_sha256=proof['initial_sha256'], seed=2026101343, batch=8,
        lr='warmup1000 to1e-5, cosine to3e-6', ema_half_life=500, snapshot_every=2000,
        feature_weight=proof['feature_weight'], auxiliary_every=4, auxiliary_coefficient=2e-5,
        auxiliary='fresh TT LF, gradients through both student stages, scalar aux zero; raw/Adam norm cap.25',
        sampler='25%wet mountains/15%oldrare/60%general; coarse original/4/2; half on-policy',
        scalar_precision='out_gain anchored FP32; embedding gains frozen for this stage',
        teacher_weights_sha256=source['teacher_weights_sha256'],
        code_sha256={name:file_sha(REPO/'distill'/name) for name in CODE}, accepted=False)
    if args.steps != 20000:
        raise ValueError('First channel-removal contract is20k; recovery is a separate reviewed stage.')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Recipe changed; preserve the previous run.')
    atomic_json(output/'contract.json', contract)
    records, _, _ = inventory()
    model = ProgressiveWidthBase(source['config'], source['target_config'], source['mapping'], source['gain_anchor']).cuda()
    model.load_state_dict(source['ema'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0., betas=(.9, .99), fused=True)
    latest = output/'latest.pt'
    step = 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
        if saved['contract'] != contract:
            raise ValueError('Resume contract changed.')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']);step = saved['step']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        torch.manual_seed(contract['seed']);torch.cuda.manual_seed_all(contract['seed'])
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    bank = BoundedQATBank(limit=96)
    variance = torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'], device='cuda')
    floors = json.loads((DATA/'eval/native-nf4-level-train-floor-4090.json').read_text())['floors']
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    started = time.monotonic()
    def save(path, full=False, raw=False):
        saved = {name:source[name] for name in ('architecture', 'config', 'target_config', 'mapping',
                 'gain_anchor', 'teacher_revision', 'teacher_weights_sha256', 'channel_calibration_sha256')}
        saved.update(step=step, amount=model.amount, contract=contract, selected_weights='raw' if raw else 'ema', accepted=False)
        if full:
            saved.update(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        else:
            saved['ema'] = {name:value.detach().cpu().to(torch.float32 if name == 'net.out_gain' else torch.bfloat16)
                            for name, value in (model if raw else ema).state_dict().items()}
        atomic_write(path, lambda handle:torch.save(saved, handle))
    if not latest.exists():
        save(output/'step000000.pt')
    try:
        while step < args.steps and not stopped:
            fraction = (step+1)/10 if args.smoke else (step+1)/args.steps
            model.amount = ema.amount = fraction*fraction*(3-2*fraction)
            lr = min(1., (step+1)/1000)*(3e-6+7e-6*.5*(1+math.cos(math.pi*max(0., (step-1000)/19000))))
            for group in optimizer.param_groups:
                group['lr'] = lr
            generator = np.random.default_rng(np.random.SeedSequence([contract['seed'], step]))
            with torch.no_grad(), model.cached_weights():
                batch, selected = bank.batch(records, generator, step, model, on_policy=step%2 == 0)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), capture_blocks(teacher) as reference:
                target = teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                                 conditional_inputs=[batch['condition'].bfloat16()]).float()
            with capture_blocks(model.net) as actual:
                prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
            loss, terms = loss_terms(prediction, target, batch, variance)
            feature = feature_loss(actual, reference, source['mapping'])
            total = loss+proof['feature_weight']*feature
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite progressive objective.')
            total.backward()
            row = dict(step=step+1, amount=model.amount, loss=float(total.detach()),
                operator=float(terms['operator'].detach()), representation=float(feature.detach()),
                lr=lr, on_policy=batch['on_policy'], auxiliary=False,
                worlds=[r['metadata']['seed'] for _, r in selected])
            del actual, reference, prediction, target, total, feature, loss, terms
            if step%4 == 3:
                regular = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                pre = preconditioners(optimizer, params)
                regular_norm, regular_scaled = gradient_norm(regular), gradient_norm(regular, pre)
                i = step//4
                kind = ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', 'wet-high-mountain')[i%5]
                coarse = ('original', 'coarse4', 'coarse2')[(i//5)%3]
                pool = [r for r in records if r['case_kind'] == kind and r['coarse_variant'] == coarse]
                record = pool[int(generator.integers(len(pool)))]
                noise_seed = int(generator.integers(1, 2**31-5821))
                fields, desired = full_level_fields(model, teacher, record, noise_seed)
                extra = level_terms(fields, desired, variance, floors[kind])['level']
                optimizer.zero_grad(set_to_none=True)
                (contract['auxiliary_coefficient']*extra).backward()
                protect_scalar_gradients(model)
                row.update(combine_gradients(params, regular, pre, regular_norm, regular_scaled, cap=.25),
                           auxiliary=True, aux_kind=kind, aux_coarse=coarse,
                           level_loss=float(extra.detach()), noise_seed=noise_seed)
                del regular, pre, fields, desired, extra
            if any(p.grad is not None for p in teacher.parameters()):
                raise ValueError('Frozen teacher received gradients.')
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1-math.exp(math.log(.5)/500))
            step += 1
            if step == 1 or step%100 == 0 or args.smoke:
                row.update(grad_norm=float(norm), elapsed_seconds=time.monotonic()-started,
                           peak_bytes=torch.cuda.max_memory_allocated())
                append_json(output/'train.jsonl', row);print(json.dumps(row), flush=True)
                atomic_json(output/'status.json', dict(status='running', step=step, target=args.steps, amount=model.amount))
            if step%2000 == 0:
                save(output/f'step{step:06d}.pt')
                save(output/'raw'/f'step{step:06d}.pt', raw=True)
            if step%1000 == 0:
                save(latest, full=True)
            if args.smoke and step >= 4:
                break
        save(latest, full=True)
        save(output/f'step{step:06d}.pt')
        atomic_json(output/'status.json', dict(status='smoke_complete' if args.smoke else 'interrupted' if stopped else 'complete',
            step=step, target=args.steps, amount=model.amount, elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        save(latest, full=True)
        atomic_json(output/'status.json', dict(status='failed', step=step, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
