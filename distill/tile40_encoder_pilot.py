"""Conditional one-pass latent encoder pilot with guided decoded-height loss.

Inputs use the frozen deployable coarse2 context and explicit two noise streams.
Stored coarse outputs are cached outputs of that same model,not hidden teacher
coarse oracles. Interior physical comparisons are a training proxy;full-tile
assembly and browser execution remain separate acceptance requirements.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from distill.common import DATA, REPO, append_json, atomic_json, atomic_write, external_path
from distill.features import base_features
from distill.native_weight_quant import file_sha
from distill.student import Student, StudentConfig, load_student
from distill.tile40_conditioning import RELIEFS
from distill.tile40_decoder_pilot import References, physical_loss, root_decode


def highpass_delta(delta):
    # Training-only crop approximation of the full1920m filter. Macro elevation
    # takes the direct sketch path;the final assembler uses the full padded tile.
    from distill.native_decoded import blur
    low = blur(F.avg_pool2d(delta, 8), 8.)
    return delta-F.interpolate(low, size=delta.shape[-2:], mode='bilinear', align_corners=False)


def fine_relief_terms(prediction_root, reference_root):
    p = prediction_root.sign()*prediction_root.square()
    t = reference_root.sign()*reference_root.square()
    losses, ratios = [], []
    for factor in (1, 4, 16):
        a, b = F.avg_pool2d(p, factor), F.avg_pool2d(t, factor)
        a = (a-F.avg_pool2d(a, 5, stride=1, padding=2))[..., 2:-2, 2:-2]
        b = (b-F.avg_pool2d(b, 5, stride=1, padding=2))[..., 2:-2, 2:-2]
        ratio = torch.sqrt((a.square().mean()+.0025)/(b.detach().square().mean()+.0025))
        losses.append((ratio-1.).square())
        ratios.append((ratio-1.).abs())
    return torch.stack(losses).mean(), torch.stack(ratios).mean()


def ridge_structure_terms(prediction_root, reference_root):
    """Match local ridge directions without requiring each fine peak's phase."""
    from distill.native_decoded import blur
    def tensor(root):
        height=root.sign()*root.square()
        background=blur(F.avg_pool2d(height,8),4.)
        high=height-F.interpolate(background,size=height.shape[-2:],mode='bilinear',align_corners=False)
        dy=F.pad(torch.diff(high,dim=-2),(0,0,0,1))
        dx=F.pad(torch.diff(high,dim=-1),(0,1,0,0))
        smooth=lambda field:F.avg_pool2d(F.pad(field,(5,5,5,5),mode='reflect'),11,stride=1)[...,16:-16,16:-16]
        aa,bb,ab=smooth(dx.square()),smooth(dy.square()),smooth(dx*dy)
        power=aa+bb
        direction=torch.cat(((aa-bb)/(power+.01),2*ab/(power+.01)),dim=1)
        coherence=torch.sqrt((aa-bb).square()+4*ab.square()+1e-8)/(power+.01)
        return direction,coherence,power
    a,ac,_=tensor(prediction_root)
    b,bc,energy=tensor(reference_root)
    weight=energy.detach()/energy.detach().mean().clamp_min(.01)
    loss=((a-b).square()*weight).mean()+.5*((ac-bc).square()*weight).mean()
    return loss,(ac-bc).abs().mean()


def encoder_geometry(y, x, halo):
    if y not in (0, 48, 96) or x not in (0, 48, 96) or halo % 32:
        raise ValueError('Original384-stride decoder origins and32-aligned encoder halo required.')
    oy, ox = ((y-halo)//32)*32, ((x-halo)//32)*32
    size = 64+2*halo+max(y-halo-oy, x-halo-ox)
    size = ((size+31)//32)*32
    return oy, ox, size, y-oy, x-ox


def encoder_window(row, y, x, halo, device):
    input_y, input_x, size, _, _ = encoder_geometry(y, x, halo)
    meta = row['metadata']
    oy, ox = meta['coarse_origin_cells']
    histogram = meta['world_config']['parameters']['histogram_raw']
    inputs = base_features(row['coarse'], oy, ox, row['case']['seed'], input_y, input_x, size, histogram)
    ly, lx = meta['latent_origin_pixels']
    target = row['latents'][:, y-ly:y-ly+64, x-lx:x-lx+64]
    if target.shape != (5, 64, 64):
        raise ValueError('Missing latent target or conditioning halo.')
    decoder_inputs, _, root = References.window(row, y*8, x*8, device)
    return inputs[None].to(device), target[None].to(device), decoder_inputs[:, :1], root


def decode(decoder, latent, gaussian):
    fields = F.interpolate(latent[:, :4], size=(512, 512), mode='nearest')
    inputs = torch.cat((gaussian, fields, fields.new_zeros(1, 11, 512, 512)), dim=1)
    with torch.autocast(latent.device.type, dtype=torch.bfloat16, enabled=latent.device.type=='cuda'):
        residual = decoder(inputs)
    return root_decode(residual, latent)


def terms(model, decoder, reference_decoder, row, y, x, variance, device, fine_weight=0., latent_weight=.2, capture=None, paired_weight=1.):
    inputs, target, noise, guided_root = encoder_window(row, y, x, model.halo, device)
    _, _, _, offset_y, offset_x = encoder_geometry(y, x, model.halo)
    with torch.no_grad():
        reference_root = decode(reference_decoder, target, noise)
    with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=inputs.device.type=='cuda'):
        prediction = model(inputs)[..., offset_y:offset_y+64, offset_x:offset_x+64]
    if capture is not None:
        capture.update(prediction=prediction, target=target,
                       conditioning=inputs[:, 10:16, offset_y:offset_y+64, offset_x:offset_x+64])
    actual_root = decode(decoder, prediction, noise)
    corrected_root = guided_root+highpass_delta(actual_root-reference_root)
    p, t = corrected_root[..., 64:-64, 64:-64], guided_root[..., 64:-64, 64:-64]
    if capture is not None:
        capture.update(height_prediction=p.sign()*p.square(), height_target=t.sign()*t.square(),
            terrain_conditioning=F.interpolate(capture['conditioning'], size=(512, 512),
                mode='nearest')[..., 128:384, 128:384])
    shape_loss, metrics = physical_loss(p, t)
    normalized = ((prediction-target).square()/variance[None, :, None, None]).mean()
    loss = paired_weight*shape_loss+latent_weight*normalized
    if fine_weight:
        fine_loss, fine_gap = fine_relief_terms(p, t)
        loss = loss+fine_weight*fine_loss
        metrics.update(fine_amplitude_gap=fine_gap)
    return loss, metrics | dict(normalized_latent=normalized)


@torch.no_grad()
def validate(model, decoder, reference_decoder, refs, variance, device, fine_weight=0., latent_weight=.2, center=False):
    model.eval()
    rows = []
    for index, row in enumerate(refs.splits['val']):
        y, x = (48, 48) if center else ((index%2)*96, ((index//2)%2)*96)
        loss, values = terms(model, decoder, reference_decoder, row, y, x, variance, device, fine_weight, latent_weight)
        rows.append(dict(case=row['case'], origin=[y, x], loss=float(loss), **{k:float(v) for k,v in values.items()}))
    keys = ['loss', 'height_mae_m', 'gradient_relative', 'amplitude_gap', 'normalized_latent']
    if fine_weight:
        keys.append('fine_amplitude_gap')
    groups = {relief:{key:float(np.mean([r[key] for r in rows if r['case']['relief']==relief])) for key in keys}
              for relief in RELIEFS}
    return dict(score=sum(g['loss'] for g in groups.values())/3., groups=groups, rows=rows,
                positions='center' if center else 'legacy peripheral positions',
                scope='Interior guided-height training proxy;cached deployable coarse2 context;not full-tile acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-coarse2-guided-v4')
    parser.add_argument('--decoder', type=Path, default=DATA/'tile40-20261014/decoder16-pilot-v1/step-020000.pt')
    parser.add_argument('--reference-decoder', type=Path, default=DATA/'final/weights/candidate-1/decoder.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--fine-weight', type=float, default=0.)
    parser.add_argument('--latent-weight', type=float, default=.2)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initial', type=Path, help='Explicit EMA initialization for a fresh data/loss contract.')
    parser.add_argument('--conditioned-blocks',action='store_true',help='Inject coarse elevation/climate into every backbone stage.')
    args = parser.parse_args()
    output = external_path(args.output)
    if (output.exists() and not args.resume) or args.steps < 1 or args.interval < 1:
        raise ValueError('Use a fresh output or explicit resume,and a positive bounded segment.')
    torch.set_num_threads(8)
    torch.manual_seed(90441)
    rng = random.Random(90441)
    device = torch.device('cuda')
    refs = References(args.dataset)
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    coarse_sha = file_sha(REPO/'distill/weights/coarse2.pt')
    if manifest['internal_coarse_checkpoint']['sha256'] != coarse_sha:
        raise ValueError('Encoder labels need the SAME deployable coarse2 input context.')
    model_class=Student
    if args.conditioned_blocks:
        from distill.tile40_conditioned import ConditionedStudent
        model_class=ConditionedStudent
    model = model_class(StudentConfig(stage='base', width=args.width, depth=args.depth)).to(device)
    if args.initial:
        source_model, _ = load_student(args.initial, device)
        if source_model.config != model.config:
            raise ValueError('Encoder initialization architecture differs from the declared pilot.')
        if args.conditioned_blocks:
            from distill.tile40_conditioned import transfer
            transfer(source_model,model)
        else:
            model.load_state_dict(source_model.state_dict())
        del source_model
    decoder, _ = load_student(args.decoder, device)
    reference_decoder, _ = load_student(args.reference_decoder, device)
    decoder.requires_grad_(False)
    reference_decoder.requires_grad_(False)
    variance = torch.stack([r['latents'].var(dim=(-2, -1), unbiased=False) for r in refs.splits['train']]).mean(0)
    variance = variance.to(device).clamp_min(.01)
    contract = dict(dataset_manifest_sha256=refs.manifest_sha, coarse_sha256=coarse_sha,
        decoder_sha256=file_sha(args.decoder), reference_decoder_sha256=file_sha(args.reference_decoder),
        width=args.width, loss_version=1, train_variance=variance.tolist(),
        scope='One-pass encoder component;final browser40km acceptance remains pending.')
    if args.depth != 3:
        contract['depth'] = args.depth
    if args.fine_weight or args.latent_weight != .2:
        contract.update(loss_version=2, fine_weight=args.fine_weight, latent_weight=args.latent_weight,
                        fine_bands_native_factors=[1, 4, 16], fine_floor_m=.05)
    if args.initial:
        contract.update(initial_sha256=file_sha(args.initial), initial_weights='ema')
    contract.update(loss_version=3,train_positions=[0,48,96],validation_positions='legacy peripheral plus separate center',
                    checkpoint_score='Mean of peripheral and central scores,equal weight.')
    if args.conditioned_blocks:
        contract.update(conditioning_architecture='coarse-film-v1',loss_version=4)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001)
    ema = {k:v.detach().clone() for k,v in model.state_dict().items()}
    step, best, stopped = 0, float('inf'), False
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract or saved['config'] != asdict(model.config):
            raise ValueError('Resume changed the dataset,decoder,variance or encoder architecture.')
        model.load_state_dict(saved['model'])
        ema = {k:v.to(device) for k,v in saved['ema'].items()}
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        step, best = saved['step'], saved['best']
    atomic_json(output/'contract.json', contract | dict(producer_sha256=file_sha(Path(__file__)),
        target_steps=args.steps, resumed_from=None if args.resume is None else str(args.resume)))
    started = time.monotonic()
    by_relief = {relief:[r for r in refs.splits['train'] if r['case']['relief']==relief] for relief in RELIEFS}
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
        if args.conditioned_blocks:
            checkpoint['conditioning_architecture']='coarse-film-v1'
        atomic_write(output/name, lambda handle:torch.save(checkpoint, handle))
    def evaluate():
        nonlocal best
        raw = validate(model, decoder, reference_decoder, refs, variance, device, args.fine_weight, args.latent_weight)
        center_raw = validate(model, decoder, reference_decoder, refs, variance, device, args.fine_weight, args.latent_weight, center=True)
        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema)
        average = validate(model, decoder, reference_decoder, refs, variance, device, args.fine_weight, args.latent_weight)
        center_ema = validate(model, decoder, reference_decoder, refs, variance, device, args.fine_weight, args.latent_weight, center=True)
        model.load_state_dict(state)
        atomic_json(output/f'validation-{step:06d}.json', dict(step=step, raw=raw, ema=average, center_raw=center_raw, center_ema=center_ema))
        raw_score, ema_score = (raw['score']+center_raw['score'])/2, (average['score']+center_ema['score'])/2
        score = min(raw_score, ema_score)
        if score < best:
            best = score
            save('best.pt')
            atomic_json(output/'best.json', dict(step=step, score=score,
                selection='raw' if raw_score <= ema_score else 'ema'))
        print(json.dumps(dict(step=step, validation=dict(raw=raw['groups'], ema=average['groups']))), flush=True)
    if step == 0:
        evaluate()
        save('step-000000.pt')
    while step < args.steps and not stopped:
        row = rng.choice(by_relief[rng.choices(RELIEFS, weights=(.6, .25, .15))[0]])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, values = terms(model, decoder, reference_decoder, row, rng.choice((0, 48, 96)), rng.choice((0, 48, 96)),
                             variance, device, args.fine_weight, args.latent_weight)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss at step{step}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        step += 1
        with torch.no_grad():
            for key,value in model.state_dict().items():
                ema[key].lerp_(value, .005)
        if step%25 == 0:
            record = dict(step=step, loss=float(loss), **{k:float(v) for k,v in values.items()},
                          elapsed_seconds=time.monotonic()-started)
            append_json(output/'training.jsonl', record)
            atomic_json(output/'status.json', dict(status='running', target_steps=args.steps, **record))
            print(json.dumps(record), flush=True)
        if step%args.interval == 0 or step == args.steps:
            evaluate()
            save(f'step-{step:06d}.pt')
    save('latest.pt')
    atomic_json(output/'status.json', dict(status='interrupted' if stopped else 'segment_complete',
        step=step, target_steps=args.steps, best=best, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
