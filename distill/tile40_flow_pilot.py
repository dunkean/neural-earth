"""Conditional latent flow pilot:learn terrain distributions instead of paired noise.

The independent Gaussian start avoids a pixel-regression compromise between
unmatched fine details. Geometry/climate still condition every velocity call.
Operator validation alone is not final-terrain or browser acceptance evidence.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import time

import torch

from distill.common import DATA, REPO, append_json, atomic_json, atomic_write, external_path
from distill.features import base_features, noise
from distill.native_weight_quant import file_sha
from distill.student import Student, StudentConfig
from distill.tile40_conditioning import RELIEFS
from distill.tile40_decoder_pilot import References
from distill.tile40_flow_runtime import coarse_altitude_guide,colored_prior,velocity,midpoint,FiniteFlowEncoder







def condition(row, y, x, size):
    meta = row['metadata']
    oy, ox = meta['coarse_origin_cells']
    inputs = base_features(row['coarse'], oy, ox, row['case']['seed'], y, x, size,
                          meta['world_config']['parameters']['histogram_raw'], include_noise=False)
    inputs[:10].zero_()
    return inputs






def midpoint_cropped(model,initial,conditioning,steps=2,schedule=None):
    """Same interior ODE result,shrinking dependency halos after each call."""
    halo=model.halo
    grid=list(schedule) if schedule is not None else [i/steps for i in range(steps+1)]
    if steps<1 or grid[0]!=0. or grid[-1]!=1. or any(b<=a for a,b in zip(grid,grid[1:])):
        raise ValueError('Increasing midpoint schedule required.')
    if min(initial.shape[-2:])<=4*steps*halo:
        raise ValueError('Missing complete iterative dependency halo.')
    crop=lambda field,margin:field[...,margin:-margin,margin:-margin] if margin else field
    state,fields=initial,conditioning
    for t,end in zip(grid,grid[1:]):
        dt=end-t
        first=crop(velocity(model,state,t,fields),halo)
        middle=crop(state,halo)+first*(dt*.5)
        second=crop(velocity(model,middle,t+dt*.5,crop(fields,halo)),halo)
        state=crop(state,2*halo)+second*dt
        fields=crop(fields,2*halo)
    import torch.nn.functional as F
    margin=2*steps*halo
    return F.pad(state,(margin,margin,margin,margin))


class FlowEncoder(torch.nn.Module):
    """Plug four tiny velocity calls into the complete tile numerical reference."""
    def __init__(self, model, contract, steps=2, split=.5):
        super().__init__()
        self.model, self.steps, self.config = model, steps, model.config
        self.colored=bool(contract.get('colored_prior'))
        self.residual_altitude=bool(contract.get('residual_altitude'))
        self.halo = 2*steps*model.halo+(32 if self.colored else 0)
        self.schedule = (0., split, 1.) if steps == 2 else None
        self.register_buffer('mean', torch.tensor(contract['latent_mean'])[None, :, None, None])
        self.register_buffer('std', torch.tensor(contract['latent_std'])[None, :, None, None])

    def forward(self, inputs):
        if self.residual_altitude:
            raise ValueError('Residual-altitude sampling requires the finite-tile boundary.')
        initial = inputs[:, :5].clone()
        if self.colored:initial=colored_prior(initial)
        conditioning = inputs.clone()
        conditioning[:, :10].zero_()
        return midpoint_cropped(self.model, initial, conditioning, self.steps, self.schedule)*self.std+self.mean




def batch(row, rng, model, mean, std, device, colored=False, centers=False):
    halo = model.halo
    positions=(32,64,96) if halo>64 else (0,32,64,96) if centers else (0,96)
    y, x = rng.choice(positions), rng.choice(positions)
    origin_y, origin_x, size = y-halo, x-halo, 64+2*halo
    ly, lx = row['metadata']['latent_origin_pixels']
    endpoint = row['latents'][:, origin_y-ly:origin_y-ly+size, origin_x-lx:origin_x-lx+size][None].to(device)
    if endpoint.shape[-2:] != (size, size):
        raise ValueError('Incomplete flow endpoint halo.')
    endpoint = (endpoint-mean)/std
    initial = noise(rng.randrange(1, 2**32), origin_y, origin_x, size, size, 5)[None].to(device)
    if colored:initial=colored_prior(initial)
    t = rng.random()
    state = initial*(1-t)+endpoint*t
    fields = condition(row, origin_y, origin_x, size)[None].to(device)
    predicted = velocity(model, state, t, fields)[..., halo:halo+64, halo:halo+64]
    target = (endpoint-initial)[..., halo:halo+64, halo:halo+64]
    return (predicted-target).square().mean()


@torch.no_grad()
def validate(model, refs, mean, std, device, colored=False, centers=False):
    model.eval()
    rng = random.Random(401923)
    rows = [dict(case=row['case'], velocity_mse=float(batch(row, rng, model, mean, std, device,colored,centers)))
            for row in refs.splits['val']]
    groups = {relief:sum(r['velocity_mse'] for r in rows if r['case']['relief']==relief)/10 for relief in RELIEFS}
    return dict(score=sum(groups.values())/3, groups=groups, rows=rows,
                scope='Independent-noise conditional flow operator validation;sampled40km quality remains unproven.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-coarse2-guided-v4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth',type=int,choices=(2,3),default=2)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--colored-prior',action='store_true',help='Give the start the measured smooth/white latent covariance structure.')
    args = parser.parse_args()
    output = external_path(args.output)
    if (output.exists() and not args.resume) or args.steps < 1 or args.interval < 1:
        raise ValueError('A fresh or explicitly resumed bounded pilot is required.')
    refs = References(args.dataset)
    plan = json.loads((args.dataset/'manifest.json').read_text())
    coarse_sha = file_sha(REPO/'distill/weights/coarse2.pt')
    if plan['internal_coarse_checkpoint']['sha256'] != coarse_sha:
        raise ValueError('Flow condition must be available from the deployed coarse model.')
    torch.set_num_threads(8)
    torch.manual_seed(107921)
    rng, device = random.Random(107921), torch.device('cuda')
    model = Student(StudentConfig(stage='base', width=args.width, depth=args.depth,
        dilations=(1,2) if args.depth==2 else (1,2,4))).to(device)
    sums = sum(row['latents'].double().sum(dim=(-2, -1)) for row in refs.splits['train'])
    squares = sum(row['latents'].double().square().sum(dim=(-2, -1)) for row in refs.splits['train'])
    count = sum(row['latents'].shape[-2]*row['latents'].shape[-1] for row in refs.splits['train'])
    mu = (sums/count).float()
    sd = (squares/count-(sums/count).square()).clamp_min(.0001).sqrt().float()
    mean, std = mu[None, :, None, None].to(device), sd[None, :, None, None].to(device)
    contract = dict(dataset_manifest_sha256=refs.manifest_sha, coarse_sha256=coarse_sha,
        latent_mean=mu.tolist(), latent_std=sd.tolist(), noise='Independent Gaussian endpoint pairing;not the teacher noise seed.',
        inference_solver='Two midpoint intervals,four velocity calls;full halo is four times model halo.',
        objective='Conditional latent distribution for decoded relief;no paired-noise pixel fidelity claim.', loss_version=1)
    if args.colored_prior:
        contract.update(colored_prior=True,prior_sigmas_latent={'0':.55,'4':4.},train_positions=[0,32,64,96],
            prior_scale='Analytic filter variance,fixed;never rescale each tile.',prior_extra_halo=32,loss_version=2)
    if args.depth!=2:
        contract.update(depth=args.depth,train_positions=[32,64,96],loss_version=3,
            rationale='Longer conditioning/noise support;complete endpoint halos;sampling cost separately measured.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.001)
    ema = {k:v.detach().clone() for k,v in model.state_dict().items()}
    step, best, stopped = 0, float('inf'), False
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract or saved['config'] != asdict(model.config):
            raise ValueError('Flow resume contract changed.')
        model.load_state_dict(saved['model'])
        ema = {k:v.to(device) for k,v in saved['ema'].items()}
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        step, best = saved['step'], saved['best']
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'contract.json', contract | dict(producer_sha256=file_sha(Path(__file__)), target_steps=args.steps))
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def save(name):
        checkpoint = dict(config=asdict(model.config), step=step, best=best, contract=contract,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            ema={k:v.detach().cpu() for k,v in ema.items()}, optimizer=optimizer.state_dict(),
            rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state())
        atomic_write(output/name, lambda handle:torch.save(checkpoint, handle))
    def evaluate():
        nonlocal best
        raw = validate(model, refs, mean, std, device,args.colored_prior,args.colored_prior)
        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema)
        average = validate(model, refs, mean, std, device,args.colored_prior,args.colored_prior)
        model.load_state_dict(state)
        atomic_json(output/f'validation-{step:06d}.json', dict(step=step, raw=raw, ema=average))
        score = min(raw['score'], average['score'])
        if score < best:
            best = score
            save('best.pt')
            atomic_json(output/'best.json', dict(step=step, score=score,
                selection='raw' if raw['score'] <= average['score'] else 'ema'))
        print(json.dumps(dict(step=step, raw=raw['groups'], ema=average['groups'])), flush=True)
    started = time.monotonic()
    if step == 0:
        evaluate()
        save('step-000000.pt')
    by_relief = {relief:[r for r in refs.splits['train'] if r['case']['relief']==relief] for relief in RELIEFS}
    while step < args.steps and not stopped:
        model.train()
        row = rng.choice(by_relief[rng.choices(RELIEFS, weights=(.6, .25, .15))[0]])
        optimizer.zero_grad(set_to_none=True)
        loss = batch(row, rng, model, mean, std, device,args.colored_prior,args.colored_prior)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite flow loss at step{step}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        step += 1
        with torch.no_grad():
            for k,v in model.state_dict().items():
                ema[k].lerp_(v, .005)
        if step%25 == 0:
            record = dict(step=step, velocity_mse=float(loss), elapsed_seconds=time.monotonic()-started)
            append_json(output/'training.jsonl', record)
            atomic_json(output/'status.json', dict(status='running', target_steps=args.steps, **record))
        if step%args.interval == 0 or step == args.steps:
            evaluate()
            save(f'step-{step:06d}.pt')
    save('latest.pt')
    atomic_json(output/'status.json', dict(status='interrupted' if stopped else 'segment_complete',
        step=step, target_steps=args.steps, best=best, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
