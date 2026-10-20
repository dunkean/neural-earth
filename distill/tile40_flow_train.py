"""Bounded finite-tile flow training, with immutable TRAIN/VAL geometry splits.

Select on sampled decoded terrain separately. Velocity loss is diagnostic only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_weight_quant import file_sha
from distill.student import StudentConfig
from distill.tile40_flow_network import TimeFlow, ARCHITECTURE
from distill.tile40_flow_pilot import colored_prior, condition, velocity,coarse_altitude_guide
from distill.tile40_conditioning import CLIMATES, RELIEFS


def climate_groups(rows):
    """Keep every TRAIN geometry's climate alternatives together."""
    groups = {relief:{} for relief in RELIEFS}
    for row in rows:
        case = row['case']
        if case['split'] != 'train':
            raise ValueError('Climate sampling only accepts TRAIN references.')
        group = groups[case['relief']].setdefault(case['seed'], {})
        if case['climate'] in group:
            raise ValueError('Duplicate climate for a TRAIN geometry.')
        group[case['climate']] = row
    if any(not seeds or any(set(group) != set(CLIMATES) for group in seeds.values())
           for seeds in groups.values()):
        raise ValueError('Every TRAIN geometry requires all five climates.')
    return {relief:list(seeds.values()) for relief,seeds in groups.items()}


def select_climate_pairs(groups, rng, batch_size, mountain_fraction):
    if batch_size < 2 or batch_size % 2 or not 0 < mountain_fraction < 1:
        raise ValueError('Even batch size and a mountain fraction between zero and one required.')
    selected = []
    remainder = 1-mountain_fraction
    for _ in range(batch_size//2):
        relief = rng.choices(RELIEFS, weights=(mountain_fraction,remainder*.625,remainder*.375))[0]
        geometry = rng.choice(groups[relief])
        selected.extend(geometry[climate] for climate in rng.sample(list(CLIMATES), 2))
    return selected


def load_rows(root):
    manifest = json.loads((root/'manifest.json').read_text())
    rows = {'train': [], 'val': []}
    for case in manifest['cases']:
        if case['split'] not in rows:
            continue
        path = root/case['file']
        if file_sha(path) != json.loads(path.with_suffix('.json').read_text())['file_sha256']:
            raise ValueError(f'Immutable reference changed: {path}')
        with np.load(path, allow_pickle=False) as item:
            metadata = json.loads(str(item['metadata']))
            if metadata['case'] != case or metadata['latent_origin_pixels'] != [-64, -64]:
                raise ValueError('Expanded latent references with complete finite-tile context required.')
            row = dict(case=case, metadata=metadata, coarse=item['coarse'].copy(),
                       latents=torch.from_numpy(item['latents'].copy()))
        rows[case['split']].append(row)
    train = {r['case']['seed'] for r in rows['train']}
    val = {r['case']['seed'] for r in rows['val']}
    test = {r['seed'] for r in manifest['cases'] if r['split'] == 'test'}
    if not train or len(rows['val']) != 30 or train & val or (train | val) & test:
        raise ValueError('Complete disjoint TRAIN, VAL and reserved TEST are required.')
    return manifest, rows


def make_batch(rows, rng, mean, std, device, size=256, augment=False,residual_altitude=False,
               paired_climates=False):
    if paired_climates:
        if len(rows)%2 or augment:
            raise ValueError('Climate pairs require an even batch without spatial augmentation.')
        for left,right in zip(rows[::2],rows[1::2]):
            a,b = left['case'],right['case']
            if (a['seed'],a['relief']) != (b['seed'],b['relief']) or a['climate']==b['climate']:
                raise ValueError('Pair the same geometry under two different climates.')
    endpoints, fields = [], []
    for index,row in enumerate(rows):
        if not paired_climates or index%2==0:
            offset_y = rng.choice(range(0, 321-size, 32))
            offset_x = rng.choice(range(0, 321-size, 32))
        endpoint = row['latents'][:, offset_y:offset_y+size, offset_x:offset_x+size].float()
        conditioning = condition(row, offset_y-64, offset_x-64, size)
        if residual_altitude:
            endpoint=endpoint.clone()
            endpoint[4:5]-=coarse_altitude_guide(conditioning[None])[0]
        if augment:
            k, flip = rng.randrange(4), rng.randrange(2)
            endpoint, conditioning = torch.rot90(endpoint, k, (-2,-1)), torch.rot90(conditioning, k, (-2,-1))
            if flip:
                endpoint, conditioning = endpoint.flip(-1), conditioning.flip(-1)
        endpoints.append(endpoint)
        fields.append(conditioning)
    endpoint = (torch.stack(endpoints).to(device)-mean)/std
    conditioning = torch.stack(fields).to(device)
    samples = len(rows)//2 if paired_climates else len(rows)
    initial = colored_prior(torch.randn(samples,*endpoint.shape[1:],device=device,dtype=endpoint.dtype))
    # Uniform times keep the underlying flow-matching objective unchanged.
    t = torch.rand(samples, 1, 1, 1, device=device)
    if paired_climates:
        initial, t = initial.repeat_interleave(2,dim=0), t.repeat_interleave(2,dim=0)
    state = initial*(1-t)+endpoint*t
    inputs = conditioning.clone()
    inputs[:, :5], inputs[:, 5:6] = state, t.expand(-1,1,size,size)
    return inputs, endpoint-initial


def loss_for(model, inputs, target):
    with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=inputs.is_cuda):
        actual = model(inputs)
    # All channels remain supervised; spatial structure gets more capacity than
    # the almost-white detail latents. Exclude crop edges from the primary loss.
    error = (actual-target)[..., 32:-32, 32:-32].square().mean(dim=(0,2,3))
    weights = error.new_tensor([2., .5, .5, .5, 2.])
    return (error*weights).sum()/weights.sum(), error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initial',type=Path,help='Fresh widened experiment initialized from an explicit half-width TimeFlow checkpoint.')
    parser.add_argument('--warm-start',type=Path,help='Explicit same-width initialization for a new representation contract.')
    parser.add_argument('--residual-altitude',action='store_true',help='Learn altitude corrections around the available coarse mean, with TRAIN residual statistics.')
    parser.add_argument('--climate-pairs',action='store_true',help='Pair two climates for one TRAIN geometry with shared crop, Gaussian start and flow time.')
    parser.add_argument('--mountain-fraction',type=float,default=.6)
    parser.add_argument('--max-inference-bytes',type=int,help='Export raw/EMA inference-only snapshots and enforce this strict byte limit.')
    parser.add_argument('--minimum-free-gib',type=float,default=20.,help='Save and stop before the active WSL volume drops below this reserve.')
    args = parser.parse_args()
    output = external_path(args.output)
    if (output.exists() and not args.resume) or min(args.steps,args.interval,args.batch_size) < 1 or (args.initial and args.warm_start) or args.minimum_free_gib < 2:
        raise ValueError('Fresh output or explicit resume and bounded positive segment required.')
    if not 0 < args.mountain_fraction < 1 or (args.climate_pairs and args.batch_size%2):
        raise ValueError('Climate pairs need an even batch;mountain fraction must be between zero and one.')
    if args.max_inference_bytes is not None and args.max_inference_bytes < 1:
        raise ValueError('Positive inference byte budget required.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable;no training has started.')
    torch.set_num_threads(8)
    torch.manual_seed(20261020)
    rng, device = random.Random(20261020), torch.device('cuda')
    manifest, rows = load_rows(args.dataset)
    coarse_sha = file_sha(REPO/'distill/weights/coarse2.pt')
    if manifest['internal_coarse_checkpoint']['sha256'] != coarse_sha:
        raise ValueError('Training requires the same coarse model available at inference.')
    count = sum(r['latents'].shape[-1]*r['latents'].shape[-2] for r in rows['train'])
    sums=torch.zeros(5,dtype=torch.float64)
    squares=torch.zeros(5,dtype=torch.float64)
    for row in rows['train']:
        latent=row['latents'].double()
        if args.residual_altitude:
            # Read the same deployable coarse field;no teacher background is
            # subtracted or supplied to the generator.
            meta=row['metadata'];oy,ox=meta['coarse_origin_cells']
            coarse=torch.from_numpy(row['coarse'][0:1,-2-oy:8-oy,-2-ox:8-ox])[None]
            guide=(torch.nn.functional.interpolate(coarse,size=(320,320),mode='bilinear',align_corners=False)+31.4)/38.6
            latent[4:5]-=guide[0].double()
        sums+=latent.sum(dim=(-2,-1))
        squares+=latent.square().sum(dim=(-2,-1))
    mu, sd = (sums/count).float(), (squares/count-(sums/count).square()).clamp_min(.0001).sqrt().float()
    mean, std = mu[None,:,None,None].to(device), sd[None,:,None,None].to(device)
    contract = dict(architecture=ARCHITECTURE, dataset_manifest_sha256=file_sha(args.dataset/'manifest.json'),
        coarse_sha256=coarse_sha, latent_mean=mu.tolist(), latent_std=sd.tolist(),
        colored_prior=True, prior_sigmas_latent={'0':.55,'4':4.}, finite_tile_boundary=True,
        crop_pixels=256, loss_margin=32, augmentation='Coordinate-aligned spatial crops only;latent channels are not assumed rotation equivariant.',
        loss_weights=[2.,.5,.5,.5,2.], noise='Independent Gaussian endpoint pairing.',
        selection='Velocity validation is diagnostic;full decoded VAL terrain selects usable checkpoints.')
    model = TimeFlow(StudentConfig(stage='base', width=args.width, depth=4)).to(device)
    if args.residual_altitude:
        contract.update(residual_altitude=True,
            altitude_guide='(bilinear(available coarse physical mean)+31.4)/38.6',
            objective='Joint detail latents and altitude correction;TRAIN residual statistics;restore guide at inference.')
    if args.climate_pairs or args.mountain_fraction != .6 or args.max_inference_bytes is not None:
        contract.update(sampling=dict(climate_pairs=args.climate_pairs,mountain_fraction=args.mountain_fraction,
            batch_size=args.batch_size,lr=args.lr,
            paired_randomness='Shared crop,Gaussian start,time;uniform distinct climate pair.' if args.climate_pairs else None),
            max_inference_bytes=args.max_inference_bytes)
    if args.warm_start:
        from distill.student import load_student
        parent,parent_saved=load_student(args.warm_start,device,ema=False)
        if parent.config!=model.config or parent_saved.get('flow_architecture')!=ARCHITECTURE or parent_saved['contract']['dataset_manifest_sha256']!=contract['dataset_manifest_sha256']:
            raise ValueError('Warm initialization requires the same architecture and immutable TRAIN bank.')
        model.load_state_dict(parent.state_dict())
        if args.residual_altitude and not parent_saved['contract'].get('residual_altitude'):
            with torch.no_grad():model.head.weight[4].zero_();model.head.bias[4].zero_()
        contract.update(warm_start_sha256=file_sha(args.warm_start),
            initialization='Same-width warm weights;reset altitude correction head when changing its representation.')
        del parent,parent_saved
    if args.initial:
        from distill.student import load_student
        from distill.tile40_flow_network import widen_time_flow
        parent,parent_saved=load_student(args.initial,device,ema=False)
        if parent_saved.get('flow_architecture')!=ARCHITECTURE or parent_saved['contract']['dataset_manifest_sha256']!=contract['dataset_manifest_sha256']:
            raise ValueError('Widening requires a matching frozen TimeFlow training bank.')
        widen_time_flow(parent,model)
        contract.update(initial_sha256=file_sha(args.initial),initialization='Balanced function-preserving twofold channel widening.',parent_step=parent_saved['step'])
        del parent,parent_saved
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001)
    ema = {k:v.detach().clone() for k,v in model.state_dict().items()}
    step, best, stopped = 0, float('inf'), False
    stop_reason=None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract or saved['config'] != asdict(model.config):
            raise ValueError('Resume contract differs;preserve this experiment.')
        model.load_state_dict(saved['model'])
        ema = {k:v.to(device) for k,v in saved['ema'].items()}
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state(saved['cuda_rng_state'])
        step, best = saved['step'], saved['best']
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'contract.json', contract | dict(target_steps=args.steps,
        parameter_bytes_fp32=sum(p.numel()*4 for p in model.parameters()), producer_sha256=file_sha(Path(__file__))))
    def stop(signum, frame):
        nonlocal stopped,stop_reason
        stopped = True
        stop_reason='signal'
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    def save(name):
        atomic_write(output/name, lambda handle:torch.save(dict(flow_architecture=ARCHITECTURE,
            config=asdict(model.config), contract=contract, step=step, best=best,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            ema={k:v.detach().cpu() for k,v in ema.items()}, optimizer=optimizer.state_dict(),
            rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state()),handle))
        if args.max_inference_bytes and (name.startswith('step-') or name=='latest.pt'):
            from distill.tile40_climate_run import export_inference
            for selection,weights in (('raw',model.state_dict()),('ema',ema)):
                export_inference(output/'inference'/f'{Path(name).stem}-{selection}.pt',
                    model.config,contract,step,weights,selection,args.max_inference_bytes)
    @torch.no_grad()
    def validate():
        nonlocal best
        # Dedicated RNGs prevent validation from changing subsequent training.
        cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.manual_seed(41917)
        vrng = random.Random(41917)
        model.eval()
        values = []
        for row in rows['val']:
            inputs, target = make_batch([row],vrng,mean,std,device,augment=False,residual_altitude=args.residual_altitude)
            loss, channels = loss_for(model,inputs,target)
            values.append(dict(case=row['case'],loss=float(loss),channels=channels.tolist()))
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(gpu_state)
        score = sum(r['loss'] for r in values)/len(values)
        atomic_json(output/f'validation-{step:06d}.json',dict(step=step,score=score,rows=values,
            scope='Velocity diagnostic;not sample quality or browser acceptance.'))
        if score < best:
            best = score
            save('best.pt')
        print(json.dumps(dict(step=step,validation_velocity=score)),flush=True)
    started = time.monotonic()
    by_relief = {kind:[r for r in rows['train'] if r['case']['relief']==kind] for kind in RELIEFS}
    paired_groups = climate_groups(rows['train']) if args.climate_pairs else None
    if step == 0:
        validate()
    while step < args.steps and not stopped:
        if step%25==0 and shutil.disk_usage(output).free < args.minimum_free_gib*2**30:
            stopped,stop_reason=True,'low-disk-space'
            break
        model.train()
        remainder = 1-args.mountain_fraction
        selected = (select_climate_pairs(paired_groups,rng,args.batch_size,args.mountain_fraction) if args.climate_pairs else
            [rng.choice(by_relief[rng.choices(RELIEFS,weights=(args.mountain_fraction,remainder*.625,remainder*.375))[0]])
             for _ in range(args.batch_size)])
        inputs, target = make_batch(selected,rng,mean,std,device,residual_altitude=args.residual_altitude,
                                    paired_climates=args.climate_pairs)
        optimizer.zero_grad(set_to_none=True)
        loss, channels = loss_for(model,inputs,target)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite finite-tile flow loss.')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        if not torch.isfinite(grad):
            raise ValueError('Nonfinite finite-tile flow gradient.')
        optimizer.step()
        step += 1
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].lerp_(value,.001)
        if step % 25 == 0:
            record = dict(step=step, loss=float(loss.detach()),channels=channels.detach().tolist(),
                grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],batch_size=args.batch_size,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
                elapsed_seconds=time.monotonic()-started)
            print(json.dumps(record),flush=True)
            append_json(output/'training.jsonl',record)
            atomic_json(output/'status.json',dict(status='running',target_steps=args.steps,**record))
        if step % args.interval == 0 or step == args.steps:
            validate()
            save(f'step-{step:06d}.pt')
    save('latest.pt')
    atomic_json(output/'status.json',dict(status='interrupted' if stopped else 'segment_complete',
        step=step,target_steps=args.steps,stop_reason=stop_reason,elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
