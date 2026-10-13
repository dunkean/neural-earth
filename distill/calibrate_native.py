"""Train-only native channel energies and unchanged-teacher reconstruction gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS, atomic_json
from distill.native_base import NativeBase, teacher_source
from distill.native_compression import compressed_config


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--output', type=Path, default=DATA/'eval/native-channel-calibration-v1.json')
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Finish native scene capture first.')
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_inference import prepare_models
    from types import SimpleNamespace
    torch.set_num_threads(8)
    # Keep ordinary version counters for the production eval cache.
    with torch.inference_mode(False):
        model = EDMUnet2D.from_pretrained(str(teacher_source())).to('cuda', torch.bfloat16).eval()
        control = NativeBase(compressed_config(model.config, model.config.model_channels, 3))
        control.net.load_state_dict({key:value.float().cpu() for key,value in model.state_dict().items()}, strict=True)
        control = control.to('cuda').eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=model, base_model=model, decoder_model=model))
    sums, counts, handles = {}, {}, []

    def observe(module, inputs, output):
        key = str(output.shape[1])
        energy = output.float().square().mean((0, 2, 3)).detach()
        sums[key] = sums.get(key, torch.zeros_like(energy))+energy
        counts[key] = counts.get(key, 0)+1

    for group in (model.enc, model.dec):
        for module in group.values():
            handles.append(module.register_forward_hook(observe))
    files = sorted(root.glob('train-*.npz'))
    chosen = np.linspace(0, len(files)-1, min(64, len(files))).round().astype(int)
    maxima, functional_maxima, mean_errors, sources = [], [], [], []
    for index in chosen:
        path = files[index]
        with np.load(path, allow_pickle=False) as sample:
            metadata = json.loads(str(sample['metadata']))
            if metadata['split'] != 'train' or metadata['seed'] in HOLDOUT_SEEDS:
                raise ValueError('Channel calibration must use training worlds only.')
            labels, call_ids = sample['labels'], sample['call_ids']
            for label in np.unique(labels):
                call_id = call_ids[np.flatnonzero(labels == label)[0]]
                selected = np.flatnonzero(call_ids == call_id)
                # Preserve original batch size: changing it can change BF16 cuDNN rounding.
                inputs, noise_labels, condition, target = [torch.from_numpy(sample[key][selected]).cuda()
                                                            for key in ('inputs', 'labels', 'condition', 'target')]
                prediction = model(inputs.to(torch.bfloat16), noise_labels=noise_labels.to(torch.bfloat16),
                                   conditional_inputs=[condition.to(torch.bfloat16)]).float()
                error = (prediction-target).abs()
                maxima.append(float(error.max()))
                mean_errors.append(float(error.mean()))
                functional = control(inputs, noise_labels, [condition])
                functional_maxima.append(float((functional-target).abs().max()))
                sources.append(dict(file=path.name, seed=metadata['seed'], rows=selected.tolist(),
                                    original_batch_size=len(selected), label=float(label)))
    for handle in handles:
        handle.remove()
    max_abs, functional_max_abs = max(maxima), max(functional_maxima)
    report = dict(status='complete', split='train', teacher_sha256=manifest['teacher_sha256'],
        dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        channel_energy={key:(value/counts[key]).cpu().tolist() for key, value in sums.items()},
        sources=sources, source_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        teacher_reconstruction=dict(max_abs=max_abs, mean_abs=float(np.mean(mean_errors)),
            functional_model_max_abs=functional_max_abs, passed=max_abs == 0. and functional_max_abs == 0.,
            note='Original call batch sizes and BF16 inputs; original cached teacher and full-width functional training model, native operator gates.'))
    atomic_json(args.output, report)
    print(json.dumps(report['teacher_reconstruction']), flush=True)
    if max_abs != 0. or functional_max_abs != 0.:
        raise RuntimeError('Uncompressed teacher did not reconstruct stored calls exactly; investigate before training.')


if __name__ == '__main__':
    main()
