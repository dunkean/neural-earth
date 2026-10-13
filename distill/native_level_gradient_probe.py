"""Read-only GPU backward probe for the new TT low-frequency objective.

No optimizer moments at EMA70k are available. Ratios here are raw-gradient
diagnostics, NOT a final Adam-preconditioned calibration or training launch.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.native_auxiliary_training import BoundedQATBank, gradient_norm, preconditioners
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_level_objective import full_level_fields, level_terms, protect_scalar_gradients
from distill.native_scalar_precision import load_scalar_precision
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_train import step_loss


def per_tensor(names, regular, auxiliary, pre=None):
    rows = []
    for i, (name, base, extra) in enumerate(zip(names, regular, auxiliary, strict=True)):
        r = float(base.float().norm()) if base is not None else 0.
        a = float(extra.float().norm()) if extra is not None else 0.
        dot = float((base.float()*extra.float()).sum()) if base is not None and extra is not None else 0.
        row = dict(name=name, regular_norm=r, auxiliary_norm=a,
                   ratio=a/r if r else None, cosine=dot/(r*a) if r*a else None)
        if pre is not None:
            rp = float((base.float()*pre[i]).norm()) if base is not None else 0.
            ap = float((extra.float()*pre[i]).norm()) if extra is not None else 0.
            row.update(regular_preconditioned_norm=rp, auxiliary_preconditioned_norm=ap,
                       preconditioned_ratio=ap/rp if rp else None)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--floors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regions', type=int, default=5, choices=(5, 32))
    parser.add_argument('--burn-in', type=int, default=0, choices=(0, 500))
    parser.add_argument('--moment-output', type=Path)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Retain prior backward evidence.')
    if args.moment_output is not None:
        external_path(args.moment_output)
        if not args.burn_in or args.moment_output.exists():
            raise ValueError('Fresh moment path and explicit zero-LR burn-in required.')
    diagnostic = json.loads((args.diagnostic/'report.json').read_text())
    floors = json.loads(args.floors.read_text())
    if (diagnostic['status'] != 'complete' or floors['status'] != 'complete' or
            floors['diagnostic_sha256'] != file_sha(args.diagnostic/'report.json')):
        raise ValueError('Complete matching TRAIN diagnostic and floors required.')
    if diagnostic['contract']['source_sha256'] != file_sha(args.source):
        raise ValueError('Common source changed.')
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    model = load_scalar_precision(saved, 'cuda').requires_grad_(True)
    anchor = float(saved['ema']['net.out_gain'].bfloat16().float())
    model.set_precision('anchored', anchor)
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_inference import prepare_models
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher, base_model=teacher, decoder_model=teacher))
    stats_path = DATA/'eval/native-decoded-probe/channel-statistics.json'
    if floors['statistics_sha256'] != file_sha(stats_path):
        raise ValueError('Channel statistics changed.')
    variance = torch.tensor(json.loads(stats_path.read_text())['variance'], device='cuda')
    records, _, _ = inventory()
    selected = diagnostic['contract']['scenes']
    if args.regions == 5:
        kinds = dict.fromkeys(r['case_kind'] for r in selected)
        selected = [next(r for r in selected if r['case_kind'] == kind) for kind in kinds]
    bank = BoundedQATBank(limit=16)
    params = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=0., betas=(.9, .99), fused=True) if args.burn_in else None
    if optimizer is not None:
        # Learn regular-objective Adam scales without modifying a single weight.
        for i in range(args.burn_in):
            step = int(saved['step'])+1000000+i
            generator = np.random.default_rng(np.random.SeedSequence([saved['contract']['seed'], step]))
            with torch.no_grad(), model.cached_weights():
                batch, _ = bank.batch(records, generator, step, model, on_policy=step%2 == 0)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = step_loss(model, teacher, batch, variance, saved['feature_weight'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if (i+1)%100 == 0:
                print(json.dumps(dict(adam_burn_in=i+1, lr=0., weight_updates=0)), flush=True)
        for name, value in model.state_dict().items():
            if not torch.equal(value, saved['ema'][name].to(device=value.device, dtype=value.dtype)):
                raise ValueError('Zero-LR Adam burn-in changed initial weights: '+name)
        optimizer.zero_grad(set_to_none=True)
    pre = preconditioners(optimizer, params) if optimizer is not None else None
    rows = []
    for index, record in enumerate(selected):
        started = time.monotonic()
        step = int(saved['step'])+100000+index
        generator = np.random.default_rng(np.random.SeedSequence([saved['contract']['seed'], step]))
        with torch.no_grad(), model.cached_weights():
            batch, _ = bank.batch(records, generator, step, model, on_policy=bool(index%2))
        loss, _, _ = step_loss(model, teacher, batch, variance, saved['feature_weight'])
        regular = torch.autograd.grad(loss, params, allow_unused=True)
        reference_norm = gradient_norm(regular)
        del loss
        model.zero_grad(set_to_none=True)
        noise_seed = int(np.random.SeedSequence([2026101322, index, 0xA0]).generate_state(1)[0])
        actual, target = full_level_fields(model, teacher, record, noise_seed)
        terms = level_terms(actual, target, variance, floors['floors'][record['case_kind']])
        if not torch.isfinite(terms['level']):
            raise FloatingPointError('Nonfinite low-frequency objective.')
        terms['level'].backward()
        scalar_before = {name: float(p.grad) for name, p in model.named_parameters()
                         if p.ndim == 0 and p.grad is not None}
        protected = protect_scalar_gradients(model)
        auxiliary = [p.grad for p in params]
        if any(p.grad is not None and p.grad.abs().sum() for p in teacher.parameters()):
            raise ValueError('Teacher received gradients.')
        if not all(g is None or torch.isfinite(g).all() for g in auxiliary):
            raise FloatingPointError('Nonfinite auxiliary gradient.')
        tensors = per_tensor(names, regular, auxiliary, pre)
        norm = gradient_norm(auxiliary)
        if reference_norm <= 0 or norm <= 0:
            raise ValueError('Active regular and auxiliary gradients required.')
        torch.cuda.synchronize()
        row = dict(index=index, kind=record['case_kind'], coarse=record['coarse_variant'],
            seed=record['metadata']['seed'], noise_seed=noise_seed, split='train',
            loss={name: float(value.detach()) for name, value in terms.items()},
            regular_norm=reference_norm, auxiliary_norm=norm, raw_ratio=norm/reference_norm,
            scalar_aux_before_protection=scalar_before, protected_scalars=protected,
            tensors=tensors, elapsed_seconds=time.monotonic()-started,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        if pre is not None:
            rp, ap = gradient_norm(regular, pre), gradient_norm(auxiliary, pre)
            row.update(regular_preconditioned_norm=rp, auxiliary_preconditioned_norm=ap,
                       preconditioned_ratio=ap/rp)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'tensors'}), flush=True)
        del actual, target, terms, regular, auxiliary
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    coefficient = .25/float(np.median([max(r['raw_ratio'], r.get('preconditioned_ratio', 0.)) for r in rows]))
    if args.moment_output is not None:
        atomic_write(args.moment_output, lambda f: torch.save(dict(
            architecture='native-nf4-level-adam-initialization', optimizer=optimizer.state_dict(),
            source_sha256=file_sha(args.source), source=str(args.source), anchor=anchor,
            parameter_names=names, adam_steps=args.burn_in, lr=0., weight_updates=0,
            precision_mode='anchored', objective='regular main only; initial A arm moments',
            code_sha256=file_sha(Path(__file__))), f))
    atomic_json(output, dict(status='complete', rows=rows, regions=len(rows), updates=0,
        source_sha256=file_sha(args.source), floors_sha256=file_sha(args.floors),
        diagnostic_sha256=file_sha(args.diagnostic/'report.json'), anchor=anchor,
        code_sha256={name: file_sha(REPO/'distill'/name) for name in
                     ('native_scalar_precision.py', 'native_level_objective.py', 'native_level_gradient_probe.py')},
        gpu=torch.cuda.get_device_name(),
        scope='Exact-anchor FP32 out_gain, fresh TT targets, both student stages differentiated; scalars protected.',
        coefficient=coefficient, adam_steps=args.burn_in, zero_lr_weights_exact=bool(args.burn_in),
        moment_output=str(args.moment_output) if args.moment_output else None,
        final_preconditioned_calibration=bool(args.burn_in and args.regions == 32), served_weights_changed=False))


if __name__ == '__main__':
    main()
