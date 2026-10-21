"""TRAIN channel calibration and GPU identity/compact/backward gates."""
import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory, WindowBank, capture_blocks
from distill.native_compression import compressed_config, project_state
from distill.native_progressive_width import ProgressiveWidthBase, compact_state, effective_kernel
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_train import loss_terms
from terrain_diffusion.models.edm_unet import EDMUnet2D


def feature_loss(actual, reference, mapping):
    terms = []
    for name, value in actual.items():
        weight = name+'.weight' if name.endswith('_conv') else name+'.conv_res1.weight'
        indices = torch.tensor(mapping[weight]['output'], device=value.device, dtype=torch.long)
        target = reference[name].detach().float().index_select(1, indices)
        selected = value.float().index_select(1, indices)
        energy = target.square().mean((0, 2, 3), keepdim=True).clamp_min(1e-4)
        terms.append(((selected-target).square()/energy).mean())
    return torch.stack(terms).mean()


def configure():
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=160)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the existing preparation.')
    output.mkdir(parents=True)
    configure()
    torch.manual_seed(2026101341)
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    config = compressed_config(dict(teacher.config), teacher.config.model_channels, 3)
    target_config = compressed_config(config, args.width, 3)
    records, root, _ = inventory()
    bank, energy, counts = WindowBank(records), {}, {}
    calibration_worlds = []
    with torch.no_grad():
        for step in range(32):
            batch, selected = bank.batch(records, np.random.default_rng(2026101341+step), step)
            with capture_blocks(teacher) as features:
                teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                        conditional_inputs=[batch['condition'].bfloat16()])
            for value in features.values():
                key = str(value.shape[1])
                score = value.float().square().mean((0, 2, 3))
                score += .25*sum(value.float().diff(dim=axis).square().mean((0, 2, 3)) for axis in (-1, -2))
                energy[key] = energy.get(key, torch.zeros_like(score))+score
                counts[key] = counts.get(key, 0)+1
            calibration_worlds += [row['metadata']['seed'] for _, row in selected]
    scores = {key:(value/counts[key]).cpu() for key, value in energy.items()}
    calibration = dict(split='train', batches=32, worlds=sorted(set(calibration_worlds)),
                       dataset_manifest_sha256=file_sha(root/'manifest.json'),
                       channel_scores={key:value.tolist() for key, value in scores.items()})
    atomic_json(output/'channel-calibration.json', calibration)
    with torch.device('meta'):
        target = EDMUnet2D.from_config(target_config)
    projected, mapping = project_state(teacher, target, scores)
    del projected, target, energy
    anchor = float(teacher.out_gain.float())
    model = ProgressiveWidthBase(config, target_config, mapping, anchor).cuda()
    model.net.load_state_dict({name:value.float() for name, value in teacher.state_dict().items()})
    saved = dict(architecture='native-progressive-width', config=config, target_config=target_config,
        mapping=mapping, gain_anchor=anchor, amount=0., step=0, teacher_revision=REVISION,
        teacher_weights_sha256=file_sha(teacher_source()/'diffusion_pytorch_model.safetensors'),
        channel_calibration_sha256=file_sha(output/'channel-calibration.json'),
        ema={name:value.detach().cpu().to(torch.float32 if name == 'net.out_gain' else torch.bfloat16)
             for name, value in model.state_dict().items()}, selected_weights='ema', accepted=False)
    atomic_write(output/'initial.pt', lambda handle:torch.save(saved, handle))
    variance = torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'], device='cuda')
    checks, ratios = [], []
    with torch.no_grad():
        for step in range(3):
            batch, _ = bank.batch(records, np.random.default_rng(8910+step), step)
            reference = teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                                conditional_inputs=[batch['condition'].bfloat16()]).float()
            actual = model(batch['inputs'], batch['labels'], [batch['condition']])
            if not torch.equal(reference, actual):
                raise ValueError('Zero-removal source identity failed.')
            checks.append(dict(step=step, max_difference=0., windows=len(actual)))
        model.amount = .15
        for step in range(16):
            batch, _ = bank.batch(records, np.random.default_rng(9900+step), step)
            with capture_blocks(teacher) as reference:
                target = teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                                 conditional_inputs=[batch['condition'].bfloat16()]).float()
            with capture_blocks(model.net) as actual:
                prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
            feature = feature_loss(actual, reference, mapping)
            _, terms = loss_terms(prediction, target, batch, variance)
            ratios.append(float(terms['weighted_operator']/feature.clamp_min(1e-8)))
    feature_weight = .25*float(np.median(ratios))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6, betas=(.9, .99), fused=True)
    backward = []
    for step, amount in enumerate((.05, .3, .8)):
        model.amount = amount
        batch, _ = bank.batch(records, np.random.default_rng(10900+step), step)
        with torch.no_grad(), capture_blocks(teacher) as reference:
            target = teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                             conditional_inputs=[batch['condition'].bfloat16()]).float()
        with capture_blocks(model.net) as actual:
            prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        loss, _ = loss_terms(prediction, target, batch, variance)
        loss = loss+feature_weight*feature_loss(actual, reference, mapping)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss) or not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):
            raise ValueError('Progressive backward failed.')
        optimizer.step()
        backward.append(dict(amount=amount, loss=float(loss.detach()), grad_norm=float(norm)))
    clone = copy.deepcopy(model)
    if clone.net._progressive_owner is not clone:
        raise ValueError('EMA owner reference does not point to the clone.')
    del clone, optimizer
    model.amount = 1.
    with torch.device('meta'):
        template = EDMUnet2D.from_config(target_config)
    dummy = EDMUnet2D.from_config(target_config)
    _, identity = project_state(dummy, template)
    del dummy, template
    compact = ProgressiveWidthBase(target_config, target_config, identity, anchor).cuda().eval().requires_grad_(False)
    compact.load_state_dict(compact_state(model))
    kernel_checks = []
    with torch.no_grad():
        for name, layer in model.net.named_modules():
            if not hasattr(layer, '_progressive_out'):
                continue
            other = compact.net.get_submodule(name)
            raw = layer.weight.bfloat16().index_select(0, layer._progressive_out).index_select(1, layer._progressive_in)
            gain = model.net.out_gain if name == 'out_conv' else model.net.get_submodule(name.rsplit('.', 1)[0]).emb_gain.bfloat16() if name.endswith('.emb_linear') else 1
            expected = effective_kernel(layer, raw, gain).bfloat16()
            actual = effective_kernel(other, other.weight.bfloat16(), gain).bfloat16()
            if not torch.equal(expected, actual):
                raise ValueError('Compact effective MP kernel changed: '+name)
            kernel_checks.append(name)
        batch, _ = bank.batch(records, np.random.default_rng(11700), 0)
        full = model(batch['inputs'], batch['labels'], [batch['condition']])
        narrow = compact(batch['inputs'], batch['labels'], [batch['condition']])
        max_difference = float((full-narrow).abs().max())
        relative_mse = float((full-narrow).square().mean()/full.square().mean().clamp_min(1e-8))
        # Changing channel dimensions selects different BF16 convolution/GEMM
        # kernels. Exact parameter semantics are checked above; this separately
        # bounds numerical forward drift rather than requiring identical GPU ops.
        if relative_mse > 1e-4 or max_difference > .125:
            raise ValueError(f'Physical channel compaction changed the forward: {max_difference}, {relative_mse}')
    atomic_json(output/'proof.json', dict(status='complete', initial_identity=checks, backward=backward,
        compact_max_difference=max_difference, compact_relative_mse=relative_mse,
        compact_effective_kernels_exact=kernel_checks,
        feature_weight=feature_weight, calibration_ratios=ratios, ema_owner_valid=True,
        initial_sha256=file_sha(output/'initial.pt'), gpu=torch.cuda.get_device_name(),
        teacher_frozen=True, target_width=args.width, blocks=3, passes=2,
        source_code_sha256={name:file_sha(REPO/'distill'/name) for name in
                           ('native_progressive_width.py', 'native_progressive_prepare.py')},
        peak_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(status='complete', identity_exact=True, compact_max_difference=max_difference,
                          compact_relative_mse=relative_mse, feature_weight=feature_weight)), flush=True)


if __name__ == '__main__':
    main()
