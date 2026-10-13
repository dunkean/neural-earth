"""Same-input TRAIN teacher experiment separating network and solver rounding."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from distill.common import HOLDOUT_SEEDS, REPO, atomic_json, external_path
from distill.native_base import teacher_source
from distill.native_decoded import DecodedRegion
from distill.native_fusion import forward_batches, window_outputs
from distill.native_grain_budget_cpu import bands, projection
from distill.native_weight_quant import file_sha


def solver_variants(inputs, velocity, labels):
    """Keep input/velocity values fixed; isolate trig and arithmetic rounding."""
    x, v, t = inputs.bfloat16(), velocity.bfloat16(), labels.bfloat16()[:, None, None, None]
    xt = (x*.5).float()
    cos_bf, sin_bf = torch.cos(t).float(), torch.sin(t).float()
    same = cos_bf*xt+sin_bf*.5*v.float()
    full = torch.cos(t.float())*xt+torch.sin(t.float())*.5*v.float()
    return dict(production=window_outputs(x, v, labels, torch.bfloat16),
                same_trig_fp32=same/.5, final_round_only=same.bfloat16().float()/.5,
                fp32_trig=full/.5)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve prior rounding budgets.')
    report = json.loads((args.diagnostic/'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Frozen complete TRAIN panel required.')
    selected = []
    for kind in dict.fromkeys(r['kind'] for r in report['rows']):
        row = next(r for r in report['rows'] if r['kind'] == kind and r['coarse'] == 'coarse4')
        selected.append((report['contract']['scenes'][row['index']], row))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_inference import prepare_models
    bf16 = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    fp32 = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().float().eval().requires_grad_(False)
    for net in (bf16, fp32):
        prepare_models(SimpleNamespace(coarse_model=net, base_model=net, decoder_model=net))
    def float_forward(inputs, labels, conditions):
        return fp32(inputs.float(), noise_labels=labels.float(),
                    conditional_inputs=[v.float() for v in conditions])
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    rows = []
    for record, row in selected:
        path = Path(record['root'])/record['file']
        if record['metadata']['split'] != 'train' or record['metadata']['seed'] in HOLDOUT_SEEDS or file_sha(path) != record['capture_sha256']:
            raise ValueError('Independent immutable TRAIN capture required.')
        region = DecodedRegion(path, 'cuda')
        region.refresh_noise(bf16, weight, row['noise_seed'])
        second = region.stages[1]
        inputs, labels, conditions = second['inputs'], second['labels'], second['condition']
        velocity = second['target'].bfloat16()
        repeat = forward_batches(bf16, inputs.bfloat16(), labels.bfloat16(), conditions.bfloat16())
        if not torch.equal(velocity, repeat):
            raise ValueError('BF16 deterministic teacher repeat differs.')
        velocity32 = forward_batches(float_forward, inputs, labels, conditions)
        repeat32 = forward_batches(float_forward, inputs, labels, conditions)
        if not torch.equal(velocity32, repeat32):
            raise ValueError('FP32 deterministic teacher repeat differs.')
        variants = solver_variants(inputs, velocity, labels)
        variants['network_fp32_production'] = window_outputs(inputs, velocity32, labels, torch.bfloat16)
        t = labels.bfloat16().float()[:, None, None, None]
        variants['network_fp32_full'] = (torch.cos(t)*(inputs.bfloat16()*.5).float()+
                                       torch.sin(t)*.5*velocity32.float())/.5
        cores = {name: region.core_crop(values, weight) for name, values in variants.items()}
        q = {name: (core[4].double()*38.6-31.4).cpu().numpy() for name, core in cores.items()}
        qb = {name: bands(values) for name, values in q.items()}
        comparisons = []
        pairs = [('production', 'same_trig_fp32'), ('final_round_only', 'same_trig_fp32'),
                 ('production', 'final_round_only'), ('fp32_trig', 'same_trig_fp32'),
                 ('network_fp32_production', 'production'), ('network_fp32_full', 'production')]
        for size in (32, 48):
            margin = (64-size)//2
            crop = lambda v: v[margin:64-margin, margin:64-margin]
            for actual, reference in pairs:
                for band in range(3):
                    actual_band, target_band = crop(qb[actual][band]), crop(qb[reference][band])
                    metrics = projection(actual_band, target_band)
                    comparisons.append(dict(size=size, band=band+1, actual=actual, reference=reference,
                        **metrics))
        rows.append(dict(kind=row['kind'], seed=row['seed'], noise_seed=row['noise_seed'], split='train',
                         record=record, comparisons=comparisons, bf16_repeat_exact=True, fp32_repeat_exact=True))
        print(json.dumps(dict(kind=row['kind'], seed=row['seed'], repeat_exact=True)), flush=True)
    atomic_json(output, dict(status='complete', rows=rows, updates=0, gpu=torch.cuda.get_device_name(),
        diagnostic_sha256=file_sha(args.diagnostic/'report.json'),
        code_sha256={name: file_sha(REPO/'distill'/name) for name in
                     ('native_rounding_budget.py', 'native_grain_budget_cpu.py', 'native_fusion.py')},
        teacher_weights_sha256=file_sha(teacher_source()/'diffusion_pytorch_model.safetensors'),
        scope='Five TRAIN captures, same BF16 teacher-first inputs for every refinement variant; no weights changed.',
        limitation='Network FP32 comparison includes FP32 weights and activations; solver effects are factorial, not assumed independent.'))


if __name__ == '__main__':
    main()
