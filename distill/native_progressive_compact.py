"""Materialize a completed channel-removal EMA and measure BF16 shape drift."""
import argparse
import json
from pathlib import Path

import torch

from distill.common import atomic_json, atomic_write, external_path
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, decoded_region_forward
from distill.native_progressive_prepare import configure
from distill.native_progressive_width import load_progressive, ProgressiveWidthBase, compact_state, effective_kernel
from distill.native_weight_quant import file_sha


def select_source(source, weights='selected', keep_master=False):
    if weights not in ('selected', 'raw', 'ema'):
        raise ValueError('Unknown source weight selection.')
    selected = dict(source)
    if weights == 'raw':
        if 'model' not in source:
            raise ValueError('Raw masters require the full training checkpoint.')
        selected.update(ema=source['model'], selected_weights='raw')
    elif weights == 'ema':
        selected['selected_weights'] = 'ema'
    if keep_master:
        body = [value for name, value in selected['ema'].items() if name.endswith('.weight')]
        if not body or any(value.dtype != torch.float32 for value in body):
            raise ValueError('True FP32 masters are required; inference snapshots cannot recover them.')
    return selected


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', choices=('selected', 'raw', 'ema'), default='selected')
    parser.add_argument('--keep-master', action='store_true', help='Retain projected FP32 masters for recovery')
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve a prior compaction.')
    configure()
    source = select_source(torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True),
                           args.weights, args.keep_master)
    if source['amount'] != 1.:
        raise ValueError('Channel removal must be completed before compaction.')
    full = load_progressive(source, 'cuda')
    identity = {name:dict(source=plan['source'],
                         output=list(range(len(plan['output']))) if plan['output'] is not None else None,
                         inputs=list(range(len(plan['inputs']))) if plan['inputs'] is not None else None)
                for name, plan in source['mapping'].items()}
    compact = ProgressiveWidthBase(source['target_config'], source['target_config'], identity, source['gain_anchor']).cuda().eval().requires_grad_(False)
    compact.load_state_dict(compact_state(full))
    kernels = []
    for name, layer in full.net.named_modules():
        if not hasattr(layer, '_progressive_out'):
            continue
        other = compact.net.get_submodule(name)
        raw = layer.weight.bfloat16().index_select(0, layer._progressive_out).index_select(1, layer._progressive_in)
        gain = full.net.out_gain if name == 'out_conv' else full.net.get_submodule(name.rsplit('.', 1)[0]).emb_gain.bfloat16() if name.endswith('.emb_linear') else 1
        if not torch.equal(effective_kernel(layer, raw, gain).bfloat16(), effective_kernel(other, other.weight.bfloat16(), gain).bfloat16()):
            raise ValueError('Compacted MP kernel changed: '+name)
        kernels.append(name)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    records, _, _ = inventory()
    rows = []
    with full.cached_weights(), compact.cached_weights():
        for kind in ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', 'wet-high-mountain'):
            record = next(row for row in records if row['case_kind'] == kind and row['coarse_variant'] == 'coarse4')
            region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
            a, b = [decoded_region_forward(model, region, weight)['core'] for model in (full, compact)]
            qa, qb = a[4].float()*38.6-31.4, b[4].float()*38.6-31.4
            ha, hb = qa.sign()*qa.square(), qb.sign()*qb.square()
            delta = (hb-ha)[16:48, 16:48]
            rows.append(dict(kind=kind, seed=record['metadata']['seed'], latent_max_difference=float((a-b).abs().max()),
                             height_mae_m=float(delta.abs().mean()), height_bias_m=float(delta.mean())))
    output.mkdir(parents=True)
    # Width-changing recovery starts a new optimizer; the wide optimizer/RNG
    # state belongs to the parent checkpoint, not the compact model payload.
    saved = {name:value for name, value in source.items()
             if name not in ('model', 'optimizer', 'rng', 'cuda_rng')}
    saved.update(config=source['target_config'], target_config=source['target_config'], mapping=identity,
        original_teacher_channel_mapping=source['mapping'], amount=0., phase='compact160',
        parent_sha256=file_sha(args.checkpoint),
        ema={name:value.detach().cpu().to(torch.float32 if name == 'net.out_gain' else torch.bfloat16)
             for name, value in compact.state_dict().items()})
    if args.keep_master:
        saved['recovery_master'] = {name:value.detach().cpu().clone() for name, value in compact.state_dict().items()}
    atomic_write(output/'step020000.pt', lambda handle:torch.save(saved, handle))
    atomic_json(output/'audit.json', dict(status='complete', kernels_exact=kernels, field_rows=rows,
        parameters=sum(parameter.numel() for parameter in compact.parameters()), gpu=torch.cuda.get_device_name(),
        source_sha256=file_sha(args.checkpoint), compact_sha256=file_sha(output/'step020000.pt'),
        selected_weights=source.get('selected_weights', 'ema'), recovery_master_retained=args.keep_master,
        note='Effective MP kernels exact; BF16 dimension-dependent numeric field drift measured separately.'))
    print(json.dumps(dict(status='complete', parameters=sum(parameter.numel() for parameter in compact.parameters()),
                         field_rows=rows)), flush=True)


if __name__ == '__main__':
    main()
