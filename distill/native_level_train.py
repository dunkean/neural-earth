"""Explicit EMA70k reinitialization: paired regular vs TT low-frequency arms."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_auxiliary_training import (BoundedQATBank, gradient_norm, preconditioners,
                                              combine_gradients)
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_level_objective import full_level_fields, level_terms, protect_scalar_gradients
from distill.native_scalar_precision import load_scalar_precision
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_train import step_loss

CODE = ('native_level_train.py', 'native_level_objective.py', 'native_scalar_precision.py',
        'native_weight_quant.py', 'native_weight_quant_train.py', 'native_fp8_training.py',
        'native_basis_training.py', 'native_decoded.py', 'native_fusion.py',
        'native_phase_loss.py', 'native_feature_kd.py')


def code_hashes():
    return {name: file_sha(REPO/'distill'/name) for name in CODE}


def configure():
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Engine:
    def __init__(self, source, calibration, floors, moments, arm, zero=False, resume=None):
        self.source, self.calibration, self.floors = source, calibration, floors
        self.arm, self.zero = arm, zero
        self.records, _, _ = inventory()
        self.model = load_scalar_precision(source, 'cuda').requires_grad_(True)
        self.anchor = float(source['ema']['net.out_gain'].bfloat16().float())
        self.model.set_precision('anchored', self.anchor)
        self.ema = copy.deepcopy(self.model).requires_grad_(False)
        for clone in (self.model, self.ema):
            if any(clone.net.get_submodule(n)._weight_quant_owner is not clone for n in clone.quantized_layers):
                raise ValueError('Quantizer owner was not retained by EMA.')
        self.params = list(self.model.parameters())
        self.names = [n for n, _ in self.model.named_parameters()]
        self.optimizer = torch.optim.Adam(self.params, lr=0., betas=(.9, .99), fused=True)
        self.step = int(source['step'])
        self.burn_completed = 500 if arm == 'a' or zero else 0
        if moments['parameter_names'] != self.names or moments['anchor'] != self.anchor:
            raise ValueError('Common zero-LR moments have a different parameter contract.')
        if resume is not None:
            self.model.load_state_dict(resume['model'])
            self.ema.load_state_dict(resume['ema'])
            self.optimizer.load_state_dict(resume['optimizer'])
            self.step = int(resume['step'])
            self.burn_completed = int(resume['burn_completed'])
        elif arm == 'a' or zero:
            self.optimizer.load_state_dict(moments['optimizer'])
        from terrain_diffusion.models.edm_unet import EDMUnet2D
        from terrain_inference import prepare_models
        self.teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
        prepare_models(SimpleNamespace(coarse_model=self.teacher, base_model=self.teacher, decoder_model=self.teacher))
        self.variance = torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'], device='cuda')
        self.bank = BoundedQATBank(limit=128)
        torch.manual_seed(2026101323)
        torch.cuda.manual_seed_all(2026101323)
        if resume is not None:
            torch.set_rng_state(resume['rng'])
            torch.cuda.set_rng_state_all(resume['cuda_rng'])

    def auxiliary_record(self, step):
        index = step//2
        kinds = ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', 'wet-high-mountain')
        kind = kinds[index%5]
        coarse = ('original', 'coarse4', 'coarse2')[(index//5)%3]
        pool = [r for r in self.records if r['case_kind'] == kind and r['coarse_variant'] == coarse]
        generator = np.random.default_rng(np.random.SeedSequence([2026101323, step, 0xA0]))
        return pool[int(generator.integers(len(pool)))], int(generator.integers(1, 2**31-5821))

    def update(self, burn_step=None):
        step = self.step if burn_step is None else burn_step
        burn = burn_step is not None
        lr = 0. if burn else 3.000001980375389e-6*min(1., (self.step-int(self.source['step'])+1)/500.)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        generator = np.random.default_rng(np.random.SeedSequence([self.source['contract']['seed'], step]))
        with torch.no_grad(), self.model.cached_weights():
            batch, selected = self.bank.batch(self.records, generator, step, self.model, on_policy=step%2 == 0)
        self.optimizer.zero_grad(set_to_none=True)
        loss, terms, _ = step_loss(self.model, self.teacher, batch, self.variance, self.source['feature_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite main objective.')
        loss.backward()
        row = dict(step=self.step+int(not burn), loss=float(loss.detach()), lr=lr,
            on_policy=batch['on_policy'], operator=float(terms['operator'].detach()),
            worlds=[r['metadata']['seed'] for _, r in selected],
            coarses=[r['coarse_variant'] for _, r in selected], auxiliary=False)
        del loss, terms
        if self.arm == 'b' and step%2 == 1:
            record, noise_seed = self.auxiliary_record(step)
            regular = [p.grad.detach().clone() if p.grad is not None else None for p in self.params]
            pre = preconditioners(self.optimizer, self.params)
            norm, scaled = gradient_norm(regular), gradient_norm(regular, pre)
            actual, target = full_level_fields(self.model, self.teacher, record, noise_seed)
            auxiliary = level_terms(actual, target, self.variance, self.floors['floors'][record['case_kind']])
            row.update(auxiliary=True, aux_kind=record['case_kind'], aux_coarse=record['coarse_variant'],
                aux_seed=record['metadata']['seed'], noise_seed=noise_seed,
                level_loss=float(auxiliary['level'].detach()))
            if not self.zero:
                self.optimizer.zero_grad(set_to_none=True)
                extra = self.calibration['coefficient']*auxiliary['level']
                if not torch.isfinite(extra):
                    raise FloatingPointError('Nonfinite level auxiliary.')
                extra.backward()
                protect_scalar_gradients(self.model)
                row.update(combine_gradients(self.params, regular, pre, norm, scaled, cap=.5))
                del extra
            del regular, pre, actual, target, auxiliary
        if any(p.grad is not None for p in self.teacher.parameters()):
            raise ValueError('Frozen teacher received gradients.')
        norm = torch.nn.utils.clip_grad_norm_(self.params, 1., error_if_nonfinite=True)
        self.optimizer.step()
        if not burn:
            with torch.no_grad():
                for a, p in zip(self.ema.parameters(), self.params):
                    a.lerp_(p, 1-math.exp(math.log(.5)/500))
            self.step += 1
        row.update(grad_norm=float(norm), peak_allocated=torch.cuda.max_memory_allocated(),
                   peak_reserved=torch.cuda.max_memory_reserved())
        return row

    def burn_in(self, should_stop=lambda: False):
        for index in range(self.burn_completed, 500):
            if should_stop():
                break
            self.update(burn_step=int(self.source['step'])+1000000+index)
            self.burn_completed = index+1
            if (index+1)%100 == 0:
                print(json.dumps(dict(arm=self.arm, burn_in=index+1, lr=0., weight_updates=0)), flush=True)
        for name, value in self.model.state_dict().items():
            if not torch.equal(value, self.source['ema'][name].to(device=value.device, dtype=value.dtype)):
                raise ValueError('B objective zero-LR burn-in changed weights: '+name)

    def save(self, path, experiment, full=False, raw=False):
        state = {name: self.source[name] for name in ('config', 'scheme', 'block', 'quantized_layers',
                 'teacher_revision', 'teacher_weights_sha256', 'contract', 'feature_weight')}
        state.update(architecture='native-weight-quant-scalar', scalar_precision='anchored-fp32-v1',
            gain_anchor=self.anchor, step=self.step, arm=self.arm, burn_completed=self.burn_completed, experiment=experiment,
            initialization='EMA70k BF16 promoted masters; new Adam, not exact 70k resume', accepted=False)
        if full:
            state.update(model=self.model.state_dict(), ema=self.ema.state_dict(),
                optimizer=self.optimizer.state_dict(), rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all())
        else:
            state['ema'] = {name: value.detach().to(dtype=torch.float32 if name == 'net.out_gain'
                else torch.bfloat16, device='cpu') for name, value in
                (self.model if raw else self.ema).state_dict().items()}
            state['selected_weights'] = 'raw' if raw else 'ema'
        atomic_write(path, lambda f: torch.save(state, f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--floors', type=Path, required=True)
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('a', 'b'), required=True)
    parser.add_argument('--end-step', type=int, required=True)
    args = parser.parse_args()
    configure()
    root = external_path(args.root)
    output = root/args.arm
    calibration, floors = json.loads(args.calibration.read_text()), json.loads(args.floors.read_text())
    source_sha = file_sha(args.source)
    if (not calibration['final_preconditioned_calibration'] or not calibration['zero_lr_weights_exact'] or
            calibration['source_sha256'] != source_sha or calibration['floors_sha256'] != file_sha(args.floors)):
        raise ValueError('Complete32-region matching Adam calibration required.')
    experiment = dict(source_sha256=source_sha, source=str(args.source), calibration_sha256=file_sha(args.calibration),
        floors_sha256=file_sha(args.floors), moments_sha256=file_sha(args.moments), code_sha256=code_hashes(),
        start=70000, target=80000, cadence=2, lambda_level=calibration['coefficient'],
        precision='anchored FP32 out_gain; all other parameter casts and activations BF16',
        objective='A regular; B regular plus fresh TT LF, both stages differentiated, scalar aux zero, cap.5',
        adam='new, regular A moments500; own-objective B burn500; zero LR then warmup500 to3e-6',
        accepted=False)
    smoke = json.loads(args.smoke.read_text())
    if smoke['status'] != 'complete' or smoke['code_sha256'] != code_hashes() or smoke['source_sha256'] != source_sha:
        raise ValueError('Matching A/B-zero and runtime GPU smoke required.')
    if (root/'experiment.json').exists() and json.loads((root/'experiment.json').read_text()) != experiment:
        raise ValueError('Experiment changed; preserve the old run.')
    if shutil.disk_usage(DATA).free < 16*2**30:
        raise ValueError('Reserve space for both full states and atomic saves.')
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    moments = torch.load(args.moments, map_location='cpu', weights_only=True, mmap=True)
    if moments['source_sha256'] != source_sha:
        raise ValueError('Moment source changed.')
    latest = output/'latest.pt'
    resume = torch.load(latest, map_location='cpu', weights_only=True, mmap=True) if latest.exists() else None
    if resume is not None and (resume['arm'] != args.arm or resume['experiment'] != experiment):
        raise ValueError('Arm resume changed.')
    if not int(source['step']) == 70000 or args.end_step > 80000:
        raise ValueError('Explicit70k to80k pilot only.')
    atomic_json(root/'experiment.json', experiment)
    engine = Engine(source, calibration, floors, moments, args.arm, resume=resume)
    if args.end_step <= engine.step:
        raise ValueError('No implicit rerun of a completed segment.')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    if engine.burn_completed < 500:
        atomic_json(output/'status.json', dict(status='burn_in', step=engine.step, weight_updates=0))
        engine.burn_in(lambda: stopped)
        if stopped:
            engine.save(latest, experiment, full=True)
            atomic_json(output/'status.json', dict(status='interrupted', step=engine.step,
                        burn_completed=engine.burn_completed, target=80000))
            return
    started = time.monotonic()
    atomic_json(output/'status.json', dict(status='running', step=engine.step, end_step=args.end_step))
    try:
        while engine.step < args.end_step and not stopped:
            row = engine.update()
            if engine.step%100 == 0 or engine.step == 70001:
                row['elapsed_seconds'] = time.monotonic()-started
                append_json(output/'train.jsonl', row)
                print(json.dumps(row), flush=True)
                atomic_json(output/'status.json', dict(status='running', step=engine.step, end_step=args.end_step))
            if engine.step%1000 == 0:
                engine.save(latest, experiment, full=True)
        engine.save(latest, experiment, full=True)
        engine.save(output/f'step{engine.step:06d}.pt', experiment)
        engine.save(output/'raw'/f'step{engine.step:06d}.pt', experiment, raw=True)
        atomic_json(output/'status.json', dict(status='interrupted' if stopped else 'segment_complete',
            step=engine.step, end_step=args.end_step, target=80000, elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        engine.save(latest, experiment, full=True)
        atomic_json(output/'status.json', dict(status='failed', step=engine.step, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
