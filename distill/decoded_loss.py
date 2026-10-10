"""Paired base/decoder supervision using existing, exactly aligned teacher crops.

Only complete decoder windows contained in the stored base target qualify.
No new teacher targets or evaluation worlds are introduced. The decoder is
frozen; gradients through its inputs supervise the base's eventual residual.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distill.common import HOLDOUT_SEEDS, atomic_json, atomic_write, external_path
from distill.dataset import Crops
from distill.features import base_features, noise
from distill.student import Student, StudentConfig, load_student


class PairedCrops(Crops):
    def __init__(self, root, split, halo, audit):
        super().__init__(root, 'base', split, halo=halo, train_size=64)
        report = json.loads(Path(audit).read_text())
        if Path(report['dataset']).resolve() != self.root.resolve() or report['mismatches']:
            raise ValueError('Paired target audit must match this dataset without latent mismatches.')
        rows = [r for r in report['rows'] if r['split'] == split]
        if not rows or any(r['seed'] in HOLDOUT_SEEDS or not r['latent_pair_exact'] for r in rows):
            raise ValueError('Paired crops must use separate training/validation worlds and exact latent pairs.')
        names = {r['file'] for r in rows}
        self.paths = [p for p in self.paths if p.name in names]
        if len(self.paths) != len(names):
            raise ValueError('Some audited paired crops are missing.')

    def __getitem__(self, address):
        index = address[0] if isinstance(address, tuple) else address
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as base, np.load(self.root/'decoder'/path.name, allow_pickle=False) as decoder:
            m, d = json.loads(str(base['metadata'])), json.loads(str(decoder['metadata']))
            if (m['y'] % 48 or m['x'] % 48 or d['y'] != 8*m['y'] or d['x'] != 8*m['x'] or
                    d['seed'] != m['seed'] or d['split'] != m['split']):
                raise ValueError('Base and complete decoder windows are not aligned.')
            target = base['target'][:, :64, :64].copy()
            if not np.array_equal(target[:4], decoder['latents']):
                raise ValueError('Teacher latent pairs changed since the alignment audit.')
            inputs = base_features(base['coarse'], m['coarse_y'], m['coarse_x'], m['seed'],
                                   m['y']-self.halo, m['x']-self.halo, 64+2*self.halo, base['histogram'])
            return (inputs, torch.from_numpy(target.astype(np.float32)),
                    torch.from_numpy(base['mask'][:, :64, :64].astype(np.float32)),
                    noise(m['seed']+5819, d['y'], d['x'], 512, 512, 1, tile=512),
                    torch.from_numpy(decoder['target'].astype(np.float32)),
                    torch.from_numpy(decoder['mask'].astype(np.float32)))


def decode(decoder, latents, gaussian):
    if decoder.config.stage != 'decoder' or latents.shape[1:] != (5, 64, 64) or gaussian.shape[1:] != (1, 512, 512):
        raise ValueError('Decoded supervision requires complete 64-latent/512-native windows.')
    fields = F.interpolate(latents[:, :4], size=(512, 512), mode='nearest')
    zeros = fields.new_zeros(fields.shape[0], 11, 512, 512)
    return decoder(torch.cat([gaussian, fields, zeros], dim=1))


def reconstructed_height(residual, latents):
    """Interior height proxy, using the pinned teacher's Laplacian reconstruction.

    Complete-window residuals have not been blended with neighbours. Exclude
    64 native pixels of boundary support; full pipeline quality is tested later.
    """
    from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
    with torch.autocast(device_type=latents.device.type, enabled=False):
        high = residual.float()*.7  # Pinned residual mean=0, std=.7.
        low = latents[:, 4:5].float()*38.6-31.4
        high, low = laplacian_denoise(high, low, sigma=5)
        sqrt_height = laplacian_decode(high, low)[..., 64:-64, 64:-64]
        return sqrt_height.sign()*sqrt_height.square()


def shore_loss(prediction, target, mask, band_m=20., temperature_m=2.):
    """Match soft land/sea probabilities near zero without changing their target.

    KL equals BCE minus the target's entropy, so an exact height prediction has
    zero loss. Each available land/sea class and each active window gets equal
    weight. Far-away relief and invalid pixels have no contribution.
    """
    if (not math.isfinite(band_m) or not math.isfinite(temperature_m) or
            min(band_m, temperature_m) <= 0):
        raise ValueError('Positive finite shore band and temperature required.')
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError('Shore height and mask shapes must agree.')
    p, t = prediction.float(), target.detach().float()
    valid = (mask > 0) & (t.abs() <= band_m)
    probability = torch.sigmoid(t/temperature_m)
    divergence = (F.binary_cross_entropy_with_logits(p/temperature_m, probability, reduction='none') -
                  F.binary_cross_entropy_with_logits(t/temperature_m, probability, reduction='none'))
    terms, present, errors = [], [], []
    for land in (t > 0, t <= 0):
        selected = (valid & land).float()
        count = selected.sum((-3, -2, -1))
        terms.append((divergence*selected).sum((-3, -2, -1))/count.clamp_min(1))
        errors.append((((p > 0) != (t > 0)).float()*selected).sum((-3, -2, -1))/count.clamp_min(1))
        present.append((count > 0).float())
    classes = torch.stack(present).sum(0)
    windows = (classes > 0).float()
    loss = (torch.stack(terms).sum(0)/classes.clamp_min(1)*windows).sum()/windows.sum().clamp_min(1)
    disagreement = (torch.stack(errors).sum(0)/classes.clamp_min(1)*windows).sum()/windows.sum().clamp_min(1)
    return loss, disagreement


def paired_loss(decoder, prediction, target, gaussian, residual, mask, spectral=.02, gradient=.05, bands=.05,
                shore_weight=0.):
    from distill.train import losses, spectral_band_loss
    decoded = decode(decoder, prediction, gaussian)
    loss, values = losses(decoded, residual, mask, spectral, gradient)
    loss += bands*spectral_band_loss(decoded, residual, mask)
    ph, th = reconstructed_height(decoded, prediction), reconstructed_height(residual, target)
    valid = mask[..., 64:-64, 64:-64]
    per_window_mae = ((ph-th).abs()*valid).sum((-2,-1))/valid.sum((-2,-1)).clamp_min(1)
    # A one-metre error on a low plain must not disappear behind mountains.
    scale = th.detach().std((-2,-1), unbiased=False).clamp_min(5.)
    relative_height = (per_window_mae/scale).mean()
    loss += .05*relative_height + bands*spectral_band_loss(ph, th, valid)
    metrics = dict(decoded_mse=values['mse'], decoded_height_mae_m_proxy=per_window_mae.mean())
    if shore_weight:
        coast, disagreement = shore_loss(ph, th, valid)
        loss += shore_weight*coast
        metrics.update(decoded_coast_kl=coast, decoded_coast_disagreement=disagreement)
    return loss, metrics | dict(decoded_loss=loss)


def frozen_decoder(path, device):
    payload = Path(path).read_bytes()
    model, saved = load_student(io.BytesIO(payload), device)
    if model.config.stage != 'decoder' or saved['step'] <= 0:
        raise ValueError('Decoded supervision requires a trained decoder checkpoint.')
    return model.eval().requires_grad_(False), dict(path=str(Path(path).resolve()),
                sha256=hashlib.sha256(payload).hexdigest(), step=saved['step'])


def initialize(checkpoint, decoder_checkpoint, audit, output, coverage=None):
    """A separate fine-tuning trial; future resumes preserve its own optimizer."""
    output = external_path(output)
    if (output/'latest.pt').exists():
        raise ValueError('A paired trial exists already; resume it instead of reinitializing.')
    payload = Path(checkpoint).read_bytes()
    source = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    config = dict(source['config'])
    config['dilations'] = tuple(config['dilations'])
    if config['stage'] != 'base' or source['step'] <= 0:
        raise ValueError('Paired fine tuning starts from a trained base.')
    torch.manual_seed(source['arguments']['seed'])
    model = Student(StudentConfig(**config))
    model.load_state_dict(source['ema'])
    paired = PairedCrops(source['arguments']['dataset'], 'train', model.halo, audit)
    policy = None
    if coverage:
        from distill.rare_cases import sampling_policy
        report = json.loads(Path(coverage).read_text())
        manifest_path = paired.root/'manifest.json'
        if report['dataset_manifest_digest'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('Rare coverage refers to a different teacher dataset.')
        names = {r['file'] for r in json.loads(Path(audit).read_text())['rows']}
        report['rows'] = [r for r in report['rows'] if r['file'] in names]
        report.pop('counts', None)
        report['note'] = 'Original low-frequency rare morphology, restricted to exact aligned teacher pairs.'
        atomic_json(output/'paired-coverage.json', report)
        sampling_policy(output/'paired-coverage.json', output/'paired-sampling.json')
        policy = json.loads((output/'paired-sampling.json').read_text())
        policy['validation_files'] = sorted(r['file'] for r in report['rows'] if r['split'] == 'val')
        atomic_json(output/'paired-sampling.json', policy)
    decoder_payload = Path(decoder_checkpoint).read_bytes()
    decoder_source = torch.load(io.BytesIO(decoder_payload), map_location='cpu', weights_only=True)
    if decoder_source['config']['stage'] != 'decoder' or decoder_source['step'] <= 0:
        raise ValueError('A trained decoder is required.')
    for name, value in [('source.pt', payload), ('decoder.pt', decoder_payload), ('paired-audit.json', Path(audit).read_bytes())]:
        atomic_write(output/name, lambda f, value=value: f.write(value))
    decoder = dict(path=str(output/'decoder.pt'), sha256=hashlib.sha256(decoder_payload).hexdigest(), step=decoder_source['step'])
    provenance = dict(source_checkpoint=str(output/'source.pt'), source_sha256=hashlib.sha256(payload).hexdigest(),
                      source_step=source['step'], source_weights='EMA', optimizer_reset=True,
                      previous_warm_start=source.get('warm_start'), paired_decoder=decoder,
                      audit_sha256=hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      note='New fine-tuning trial on existing aligned teacher pairs; no reserved evaluation seed is trained.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4, fused=True)
    state = dict(source, config=asdict(model.config), model=model.state_dict(), ema=model.state_dict(),
                 optimizer=optimizer.state_dict(), step=0, best_score=float('inf'), validation=None,
                 train_files=[p.name for p in paired.paths], sampling_policy=policy, rng=torch.get_rng_state(),
                 paired_decoder=decoder, paired_audit=provenance['audit_sha256'], warm_start=provenance)
    state.pop('cuda_rng', None)
    state['arguments'] = dict(source['arguments'], output=str(output), train_size=64,
                             paired_decoder=decoder['path'], paired_audit=str(output/'paired-audit.json'), decoder_loss_weight=1.)
    atomic_write(output/'latest.pt', lambda f: torch.save(state, f))
    atomic_json(output/'warm-start.json', provenance)
    atomic_json(output/'status.json', dict(status='initialized', step=0, accepted=False))
    return provenance


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('decoder', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--coverage', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(initialize(args.checkpoint, args.decoder, args.audit, args.output, args.coverage)), flush=True)


if __name__ == '__main__':
    main()
