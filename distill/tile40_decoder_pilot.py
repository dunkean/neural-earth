"""Train a thin decoder component on guided40km references, with physical loss.

Teacher latents remain oracle inputs in this component experiment. This is
neither a sketch-to-tile product nor a full-tile latency/quality benchmark.
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

from distill.common import DATA, atomic_json, atomic_write, append_json, external_path
from distill.features import decoder_features
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_conditioning import RELIEFS, sample_sketch
from distill.tile40_decoder_projection import project_decoder


def root_decode(residual, latents):
    from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
    with torch.autocast(device_type=residual.device.type, enabled=False):
        high, low = laplacian_denoise(residual.float()*.7, latents[:, 4:5].float()*38.6-31.4, sigma=5)
        return laplacian_decode(high, low)[..., 64:-64, 64:-64]


def physical_loss(prediction_root, reference_root):
    """Position, slope and band amplitude errors in the FINAL guided height."""
    p = prediction_root.sign()*prediction_root.square()
    t = reference_root.sign()*reference_root.square()
    scale = t.detach().std(unbiased=False).clamp_min(20.)
    mae = (p-t).abs().mean()
    gradients = []
    for axis in (-2, -1):
        dp, dt = torch.diff(p, dim=axis), torch.diff(t, dim=axis)
        gradients.append((dp-dt).square().mean()/dt.detach().square().mean().clamp_min(1.))
    amplitude, ratios = [], []
    for factor in (4, 16, 32):
        a, b = F.avg_pool2d(p, factor), F.avg_pool2d(t, factor)
        a = a-F.avg_pool2d(a, 3, stride=1, padding=1)
        b = b-F.avg_pool2d(b, 3, stride=1, padding=1)
        a, b = a[..., 1:-1, 1:-1], b[..., 1:-1, 1:-1]
        aa, bb = a.square().mean(), b.square().mean()
        # A metre floor controls nearly-flat cases without an unstable ratio.
        amplitude.append((torch.sqrt(aa+1.)-torch.sqrt(bb+1.)).square()/(bb+1.))
        ratios.append(torch.sqrt((aa+1.)/(bb+1.)))
    gradient = torch.stack(gradients).mean()
    band_loss = torch.stack(amplitude).mean()
    return mae/scale+.2*gradient+.2*band_loss, dict(height_mae_m=mae,
        gradient_relative=gradient, amplitude_gap=torch.stack([(r-1.).abs() for r in ratios]).mean())


class References:
    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        if manifest['target'] != 'sketch-root+teacher-decoded-highpass-root':
            raise ValueError('Guided references required.')
        self.splits = {'train': [], 'val': []}
        self.manifest_sha = file_sha(self.root/'manifest.json')
        coordinates = (np.arange(1344)+.5)*30.
        for case in manifest['cases']:
            if case['split'] not in self.splits:
                continue
            path = self.root/case['file']
            sidecar = json.loads(path.with_suffix('.json').read_text())
            if file_sha(path) != sidecar['file_sha256']:
                raise ValueError(f'Reference changed: {path}')
            with np.load(path, allow_pickle=False) as item:
                meta = json.loads(str(item['metadata']))
                if meta['case'] != case or meta['latent_origin_pixels'] not in ([-32, -32], [-64, -64]):
                    raise ValueError('Reference coordinates or case changed.')
                background = sample_sketch(item['sketch_m'], coordinates, coordinates)
                root = np.sign(background)*np.sqrt(np.abs(background))+item['root_detail_native']
                row = dict(case=case, latents=torch.from_numpy(item['latents'].astype(np.float32)),
                           guided_root=torch.from_numpy(root.astype(np.float32)), metadata=meta,
                           coarse=item['coarse'].copy())
            self.splits[case['split']].append(row)
        train_seeds = {r['case']['seed'] for r in self.splits['train']}
        val_seeds = {r['case']['seed'] for r in self.splits['val']}
        if not train_seeds or not val_seeds or train_seeds & val_seeds:
            raise ValueError('Separate complete training/validation references required.')
        expected_train = sum(case['split']=='train' for case in manifest['cases'])
        if expected_train < 120:
            raise ValueError('At least120 declared independent TRAIN cases required.')
        test_seeds = {case['seed'] for case in manifest['cases'] if case['split']=='test'}
        if test_seeds & (train_seeds | val_seeds):
            raise ValueError('Reserved TEST seed entered training or validation.')
        for split, expected in [('train', expected_train), ('val', 30)]:
            if len(self.splits[split]) != expected:
                raise ValueError(f'Incomplete {split} split.')

    @staticmethod
    def window(row, y, x, device):
        if y not in (0, 384, 768) or x not in (0, 384, 768):
            raise ValueError('Original384-stride complete512px windows required.')
        origin_y, origin_x = row['metadata']['latent_origin_pixels']
        yy, xx = y//8-origin_y, x//8-origin_x
        latents = row['latents'][:, yy:yy+64, xx:xx+64]
        inputs = decoder_features(latents[:4], row['case']['seed'], y, x)
        root = row['guided_root'][y+64:y+448, x+64:x+448]
        return inputs[None].to(device), latents[None].to(device), root[None, None].to(device)


@torch.no_grad()
def teacher_forward(teacher, inputs, latents, original=False):
    return teacher(latents, inputs[:, :1]) if original else teacher(inputs)


def validate(model, teacher, refs, device, original=False, fine_weight=0.):
    model.eval()
    rows = []
    for i, row in enumerate(refs.splits['val']):
        # All30 cases, including all5 climates; positions cover all9 windows.
        y, x = (i%3)*384, ((i//3)%3)*384
        inputs, latents, target_root = refs.window(row, y, x, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, t = model(inputs), teacher_forward(teacher, inputs, latents, original)
        corrected_root = target_root+root_decode(p, latents)-root_decode(t, latents)
        loss, values = physical_loss(corrected_root, target_root)
        if fine_weight:
            from distill.tile40_encoder_pilot import fine_relief_terms
            fine_loss, fine_gap = fine_relief_terms(corrected_root, target_root)
            loss = loss+fine_weight*fine_loss
            values['fine_amplitude_gap'] = fine_gap
        rows.append(dict(case=row['case'], origin=[y, x], loss=float(loss),
                         residual_mse=float((p-t).square().mean()), **{k:float(v) for k,v in values.items()}))
    keys = ['loss', 'residual_mse', 'height_mae_m', 'gradient_relative', 'amplitude_gap']
    if fine_weight:
        keys.append('fine_amplitude_gap')
    groups = {relief:{key:float(np.mean([row[key] for row in rows if row['case']['relief']==relief])) for key in keys}
              for relief in RELIEFS}
    score = sum(groups[relief]['loss'] for relief in RELIEFS)/3.
    return dict(score=score, groups=groups, rows=rows,
                scope='Decoder component with oracle teacher latents and fixed guided-root context;not sketch-to-tile.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-guided-v2')
    parser.add_argument('--teacher', type=Path, default=DATA/'final/weights/candidate-1/decoder.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--original-teacher', action='store_true')
    parser.add_argument('--initial', type=Path, help='Explicit trained initialization for direct-original refinement.')
    parser.add_argument('--fine-weight', type=float, default=0.)
    args = parser.parse_args()
    if args.steps < 1 or args.interval < 1:
        raise ValueError('Positive finite segment required.')
    output = external_path(args.output)
    if output.exists() and not args.resume:
        raise ValueError('Preserve old pilot artifacts;use --resume or a fresh output.')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(70401)
    rng = random.Random(70401)
    device = torch.device('cuda')
    refs = References(args.dataset)
    source, _ = load_student(args.initial or args.teacher, 'cpu')
    student = project_decoder(source, args.width).to(device)
    if args.original_teacher:
        from distill.native_decoded import OriginalDecoder
        from distill.native_base import teacher_source
        teacher = OriginalDecoder(device)
        teacher_sha = file_sha(teacher_source().parent/'decoder_model/diffusion_pytorch_model.safetensors')
    else:
        teacher, _ = load_student(args.teacher, device)
        teacher_sha = file_sha(args.teacher)
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=.001)
    ema = {k:v.detach().clone() for k,v in student.state_dict().items()}
    step, best = 0, float('inf')
    contract = dict(dataset_manifest_sha256=refs.manifest_sha, teacher_sha256=teacher_sha,
        width=args.width, loss_version=1, sample_weights=dict(mountains=.6, hills=.25, plain=.15),
        scope='Oracle-latent decoder distillation only;no deployable sketch-to-tile claim.')
    if args.original_teacher or args.initial or args.fine_weight:
        contract.update(loss_version=2, original_teacher=args.original_teacher, fine_weight=args.fine_weight,
                        initial_sha256=None if args.initial is None else file_sha(args.initial))
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('Resume contract changed.')
        student.load_state_dict(saved['model'])
        ema = {k:v.to(device) for k,v in saved['ema'].items()}
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        step, best = saved['step'], saved['best']
    atomic_json(output/'contract.json', contract | dict(arguments={k:str(v) if isinstance(v, Path) else v
        for k,v in vars(args).items()}, producer_sha256=file_sha(Path(__file__))))
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    by_relief = {relief:[r for r in refs.splits['train'] if r['case']['relief']==relief] for relief in RELIEFS}

    def save_checkpoint(name):
        saved = dict(config=asdict(student.config), step=step, best=best, contract=contract,
            model={k:v.detach().cpu() for k,v in student.state_dict().items()},
            ema={k:v.detach().cpu() for k,v in ema.items()}, optimizer=optimizer.state_dict(),
            rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state())
        atomic_write(output/name, lambda handle:torch.save(saved, handle))

    def evaluate():
        nonlocal best
        raw = validate(student, teacher, refs, device, args.original_teacher, args.fine_weight)
        state = {k:v.detach().clone() for k,v in student.state_dict().items()}
        student.load_state_dict(ema)
        average = validate(student, teacher, refs, device, args.original_teacher, args.fine_weight)
        student.load_state_dict(state)
        report = dict(step=step, raw=raw, ema=average)
        atomic_json(output/f'validation-{step:06d}.json', report)
        score = min(raw['score'], average['score'])
        if score < best:
            best = score
            save_checkpoint('best.pt')
            atomic_json(output/'best.json', dict(step=step, score=score,
                selection='raw' if raw['score'] <= average['score'] else 'ema'))
        print(json.dumps(dict(step=step, validation=dict(raw=raw['groups'], ema=average['groups']))), flush=True)

    started = time.monotonic()
    if step == 0:
        evaluate()
        save_checkpoint('step-000000.pt')
    while step < args.steps and not stopped:
        row = rng.choice(by_relief[rng.choices(RELIEFS, weights=(.6, .25, .15))[0]])
        y, x = rng.choice((0, 384, 768)), rng.choice((0, 384, 768))
        inputs, latents, target_root = refs.window(row, y, x, device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            reference = teacher_forward(teacher, inputs, latents, args.original_teacher)
        with torch.no_grad():
            reference_root = root_decode(reference, latents)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = student(inputs)
        corrected_root = target_root+root_decode(prediction, latents)-reference_root
        shape_loss, metrics = physical_loss(corrected_root, target_root)
        if args.fine_weight:
            from distill.tile40_encoder_pilot import fine_relief_terms
            fine_loss, fine_gap = fine_relief_terms(corrected_root, target_root)
            shape_loss = shape_loss+args.fine_weight*fine_loss
            metrics['fine_amplitude_gap'] = fine_gap
        residual = (prediction-reference).square().mean()/reference.square().mean().clamp_min(.05)
        loss = residual+.5*shape_loss
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at step{step}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.)
        optimizer.step()
        step += 1
        with torch.no_grad():
            for key,value in student.state_dict().items():
                ema[key].lerp_(value, .005)
        if step%25 == 0:
            record = dict(step=step, loss=float(loss), residual=float(residual),
                **{k:float(v) for k,v in metrics.items()}, elapsed_seconds=time.monotonic()-started)
            append_json(output/'training.jsonl', record)
            atomic_json(output/'status.json', dict(status='running', **record, target_steps=args.steps))
            print(json.dumps(record), flush=True)
        if step%args.interval == 0 or step == args.steps:
            evaluate()
            save_checkpoint(f'step-{step:06d}.pt')
    save_checkpoint('latest.pt')
    atomic_json(output/'status.json', dict(status='interrupted' if stopped else 'segment_complete',
        step=step, target_steps=args.steps, best=best, elapsed_seconds=time.monotonic()-started,
        component_only=True))


if __name__ == '__main__':
    main()
