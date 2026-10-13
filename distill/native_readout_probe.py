"""Test whether frozen native features can recover the critical teacher channels.

TRAIN-only ridge readouts add residuals to first-phase channel0 or second-phase
H4. Other phase/channel outputs remain exactly those of the frozen source.
This diagnostic is not an accepted runtime model or a performance result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write
from distill.native_base import NativeBase
from distill.native_decoded import DecodedRegion
from distill.native_feature_train import balanced_windows
from distill.native_multilod_eval import validation_records
from distill.native_phase_loss import PHASE_LABELS


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class RidgeStatistics:
    def __init__(self, dimensions, device):
        self.gram = torch.zeros(dimensions+1, dimensions+1, dtype=torch.float64, device=device)
        self.rhs = torch.zeros(dimensions+1, dtype=torch.float64, device=device)
        self.target_square = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0

    def add(self, features, residual):
        x = F.pad(features.detach().double(), (0, 1), value=1.)
        y = residual.detach().double()
        self.gram += x.T@x
        self.rhs += x.T@y
        self.target_square += y.square().sum()
        self.count += len(x)

    def solve(self, ridge):
        if self.count == 0 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError('Positive fixed ridge and nonempty TRAIN samples required.')
        gram = self.gram/self.count
        penalty = gram[:-1, :-1].diagonal().mean().clamp_min(1e-8)*ridge
        regularizer = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)*penalty
        regularizer[-1, -1] = 0.
        coefficients = torch.linalg.solve(gram+regularizer, self.rhs/self.count)
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError('Nonfinite ridge coefficients.')
        return coefficients.float()

    def mse(self, coefficients):
        c = coefficients.double()
        return float((self.target_square-2*c@self.rhs+c@self.gram@c)/self.count)


def combine_statistics(items):
    result = RidgeStatistics(len(items[0].rhs)-1, items[0].rhs.device)
    for item in items:
        result.gram += item.gram; result.rhs += item.rhs
        result.target_square += item.target_square; result.count += item.count
    return result


def world_folds(records):
    seeds = sorted({r['metadata']['seed'] for r in records})
    if len(seeds) < 3:
        raise ValueError('At least three independent TRAIN worlds required.')
    np.random.default_rng(17002).shuffle(seeds)
    return {seed: index % 3 for index, seed in enumerate(seeds)}


def apply_readouts(prediction, hidden, labels, coefficients, mode):
    correction = F.conv2d(hidden.float(), coefficients[:, :-1].reshape(2, hidden.shape[1], 3, 3),
                          bias=coefficients[:, -1], padding=1)
    output = prediction.clone()
    if mode in ('first', 'both'):
        mask = torch.isclose(labels.float(), labels.new_tensor(PHASE_LABELS[0]).float())
        output[:, 0] += correction[:, 0]*mask[:, None, None]
    if mode in ('second', 'both'):
        mask = torch.isclose(labels.float(), labels.new_tensor(PHASE_LABELS[1]).float())
        output[:, 4] += correction[:, 1]*mask[:, None, None]
    return output


class FrozenFeatures:
    def __init__(self, source, kind):
        self.kind, self.values = kind, {}
        self.handles = [source.net.out_conv.register_forward_pre_hook(self.last)]
        self.names = ['out_conv:input']
        if kind == 'multiscale':
            prefix = next(iter(source.net.enc)).split('_')[0]
            high = [name for name in source.net.enc if name.startswith(prefix+'_block')][-1]
            middle = next(name for name in source.net.dec if name.endswith('_in1'))
            self.names += ['enc.'+high, 'dec.'+middle]
            self.handles += [source.net.enc[high].register_forward_hook(self.high),
                             source.net.dec[middle].register_forward_hook(self.middle)]

    def last(self, module, args):
        self.values['last'] = args[0]

    def high(self, module, args, output):
        self.values['high'] = output

    def middle(self, module, args, output):
        self.values['middle'] = output

    def patches(self):
        last = self.values['last'].float()
        pieces = [F.unfold(last, 3, padding=1)]
        if self.kind == 'multiscale':
            for key in ('high', 'middle'):
                values = self.values[key].float()
                if values.shape[-2:] != last.shape[-2:]:
                    values = F.interpolate(values, size=last.shape[-2:], mode='bilinear', align_corners=False)
                pieces.append(values.flatten(2))
        return torch.cat(pieces, dim=1)

    def remove(self):
        for handle in self.handles:
            handle.remove()


def apply_feature_readouts(prediction, patches, labels, coefficients, mode):
    correction = torch.einsum('bdn,kd->bkn', patches, coefficients[:, :-1])
    correction += coefficients[:, -1][None, :, None]
    correction = correction.reshape(len(prediction), 2, *prediction.shape[-2:])
    output = prediction.clone()
    for phase, channel, enabled in ((0, 0, mode in ('first', 'both')), (1, 4, mode in ('second', 'both'))):
        if enabled:
            mask = torch.isclose(labels.float(), labels.new_tensor(PHASE_LABELS[phase]).float())
            output[:, channel] += correction[:, phase]*mask[:, None, None]
    return output


class ReadoutProbe(nn.Module):
    def __init__(self, source, coefficients, mode, features='last'):
        super().__init__()
        self.source, self.mode, self.config = source, mode, source.config
        self.register_buffer('coefficients', coefficients)
        self.features = FrozenFeatures(source, features)

    def forward(self, inputs, labels, conditional_inputs):
        prediction = self.source(inputs, labels, conditional_inputs)
        if self.features.kind == 'last':
            return apply_readouts(prediction, self.features.values['last'], labels, self.coefficients, self.mode)
        return apply_feature_readouts(prediction, self.features.patches(), labels, self.coefficients, self.mode)


def load_source(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('teacher_revision') or saved.get('dataset_manifest', {}).get('revision')) != REVISION:
        raise ValueError('Source teacher revision differs.')
    source = NativeBase(saved['config']).cuda().eval().requires_grad_(False)
    source.load_state_dict(saved['ema'])
    return source, saved['step']


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit', 'operator'))
    parser.add_argument('--source', type=Path, default=DATA/'ckpt/native160x3-operator-pilot/step15000.pt')
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ridge', type=float, default=.01)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--features', choices=('last', 'multiscale'), default='last')
    args = parser.parse_args()
    root = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((root/'manifest.json').read_text())
    stats = json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())
    records = validation_records(manifest, stats)
    calibration_path = DATA/'eval/native-phase-train-band-statistics.json'
    calibration = json.loads(calibration_path.read_text())
    if calibration['split'] != 'train' or calibration['dataset_manifest_sha256'] != digest(root/'manifest.json'):
        raise ValueError('Matching TRAIN calibration inventory required.')
    source, step = load_source(args.source)
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    capture = FrozenFeatures(source, args.features)
    contract = dict(source=str(args.source.resolve()), source_sha256=digest(args.source), source_step=step,
        teacher_revision=REVISION, dataset_manifest_sha256=digest(root/'manifest.json'),
        calibration_sha256=digest(calibration_path), producer_sha256=digest(Path(__file__)),
        ridge=args.ridge, ridge_grid=[.0001, .001, .01, .1, 1.], fit_pixels_per_window=64, seed=17001,
        features=args.features, feature_modules=capture.names,
        channels=['first:0', 'second:4'], smoke=args.smoke, accepted=False)
    if args.action == 'fit':
        if args.probe.exists():
            raise ValueError('Do not overwrite a fitted diagnostic.')
        selected = set(calibration['scene_files'])
        records = [r for r in manifest['scenes'] if r['metadata']['split'] == 'train' and r['file'] in selected]
        if len(records) != len(selected):
            raise ValueError('Calibration contains non-TRAIN files.')
        if args.smoke:
            records = records[:3]
        folds = world_folds(records) if not args.smoke else {r['metadata']['seed']: 0 for r in records}
        accumulators = sanity = None
        for index, record in enumerate(records):
            generator = np.random.default_rng(17001+index)
            batch = balanced_windows(DecodedRegion(root/record['file'], 'cuda'), generator)
            prediction = source(batch['inputs'], batch['labels'], [batch['condition']])
            patches = capture.patches().transpose(1, 2)
            if accumulators is None:
                dimensions = patches.shape[-1]
                accumulators = [[RidgeStatistics(dimensions, 'cuda') for _ in range(3)] for _ in range(2)]
                sanity = RidgeStatistics(dimensions, 'cuda')
            eligible = np.arange(64*64).reshape(64, 64)[1:-1, 1:-1].ravel()
            for phase, channel in enumerate((0, 4)):
                for window in range(2*phase, 2*phase+2):
                    ids = torch.tensor(generator.choice(eligible, 64, replace=False), device='cuda')
                    residual = (batch['target'][window, channel]-prediction[window, channel]).flatten()
                    accumulators[phase][folds[record['metadata']['seed']]].add(patches[window, ids], residual[ids])
                    if phase == 0:
                        sanity.add(patches[window, ids], prediction[window, channel].flatten()[ids])
            if index % 25 == 0:
                print(json.dumps(dict(completed=index+1, total=len(records))), flush=True)
        grid = [.0001, .001, .01, .1, 1.]
        selected_ridges, cv_scores, fitted = [], [], []
        for phase in range(2):
            if args.smoke:
                scores = {}; selected_ridge = args.ridge
            else:
                scores = {}
                for ridge in grid:
                    errors = []
                    for fold in range(3):
                        training = combine_statistics([a for index, a in enumerate(accumulators[phase]) if index != fold])
                        coefficients = training.solve(ridge)
                        errors.append(accumulators[phase][fold].mse(coefficients))
                    scores[str(ridge)] = errors
                selected_ridge = min(grid, key=lambda ridge: np.mean(scores[str(ridge)]))
            selected_ridges.append(selected_ridge); cv_scores.append(scores)
            fitted.append(combine_statistics(accumulators[phase]).solve(selected_ridge))
        coefficients = torch.stack(fitted).cpu()
        atomic_write(args.probe, lambda f: torch.save(dict(coefficients=coefficients, contract=contract,
            sanity_coefficients=sanity.solve(1e-6).cpu(), world_folds=folds,
            selected_ridges=selected_ridges, cv_scores=cv_scores,
            train_files=[r['file'] for r in records],
            samples_per_phase=[sum(a.count for a in group) for group in accumulators]), f))
        atomic_json(args.probe.with_suffix('.json'), dict(status='complete', **contract,
            gpu=torch.cuda.get_device_name(), train_scenes=len(records), probe_sha256=digest(args.probe)))
    else:
        if args.output is None:
            parser.error('--output required for operator validation')
        saved = torch.load(args.probe, map_location='cpu', weights_only=True)
        if saved['contract'] != contract or saved['contract']['smoke']:
            raise ValueError('Only the exact completed full TRAIN probe can be validated.')
        coefficients = saved['coefficients'].cuda(); rows = []; sanity_errors = []; sanity_targets = []
        for index, record in enumerate(records):
            batch = balanced_windows(DecodedRegion(root/record['file'], 'cuda'), np.random.default_rng(17001+index))
            prediction = source(batch['inputs'], batch['labels'], [batch['condition']])
            patches = capture.patches()
            candidate = apply_feature_readouts(prediction, patches, batch['labels'], coefficients, 'both')
            sanity_coefficients = saved['sanity_coefficients'].cuda()
            sanity_prediction = (torch.einsum('bdn,d->bn', patches[:2], sanity_coefficients[:-1])
                                 +sanity_coefficients[-1]).reshape_as(prediction[:2, 0])
            sanity_errors.append(float((sanity_prediction-prediction[:2, 0]).square().mean()))
            sanity_targets.append(float(prediction[:2, 0].square().mean()))
            for phase, channel in enumerate((0, 4)):
                section = slice(2*phase, 2*phase+2)
                rows.append(dict(file=record['file'], coarse_variant=record['coarse_variant'], phase=phase,
                    source_mse=float((prediction[section, channel]-batch['target'][section, channel]).square().mean()),
                    probe_mse=float((candidate[section, channel]-batch['target'][section, channel]).square().mean())))
        groups = {}
        for variant in ('original', 'coarse4', 'coarse2'):
            groups[variant] = {}
            for phase in (0, 1):
                subset = [r for r in rows if r['coarse_variant'] == variant and r['phase'] == phase]
                groups[variant][str(phase)] = {key: float(np.mean([r[key] for r in subset]))
                                               for key in ('source_mse', 'probe_mse')}
        atomic_json(args.output, dict(status='complete', contract=contract, groups=groups, rows=rows,
            sanity_relative_mse=float(np.mean(sanity_errors)/np.mean(sanity_targets)),
            selected_ridges=saved['selected_ridges'], cv_scores=saved['cv_scores'],
            probe_sha256=digest(args.probe), gpu=torch.cuda.get_device_name(), accepted=False))
        print(json.dumps(groups), flush=True)
    capture.remove()


if __name__ == '__main__':
    main()
