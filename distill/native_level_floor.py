"""TRAIN precision floors for the isolated low-frequency objective.

Only base arithmetic changes to FP32: solver/fusion/noise and original decoder
remain BF16, matching the existing fp32base reference. No student fitting.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import DATA, REPO, HOLDOUT_SEEDS, atomic_json, external_path
from distill.decoded_loss import reconstructed_height
from distill.native_base import teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder, blur
from distill.native_field_auxiliary import low_height
from distill.native_level_diagnostic import height_stats
from distill.native_weight_quant import file_sha


class FP32Teacher(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, noise_labels, conditional_inputs):
        return self.model(inputs.float(), noise_labels=noise_labels.float(),
                          conditional_inputs=[v.float() for v in conditional_inputs]).to(inputs.dtype)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve previous precision floors.')
    diagnostic = json.loads((args.diagnostic/'report.json').read_text())
    if diagnostic['status'] != 'complete' or diagnostic['contract']['split'] != 'train':
        raise ValueError('Complete TRAIN attribution required.')
    contract = diagnostic['contract']
    stats_path = DATA/'eval/native-decoded-probe/channel-statistics.json'
    variance = torch.tensor(json.loads(stats_path.read_text())['variance'], device='cuda')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_inference import prepare_models
    teacher_bf16 = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    teacher_fp32 = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().float().eval().requires_grad_(False)
    for teacher in (teacher_bf16, teacher_fp32):
        prepare_models(SimpleNamespace(coarse_model=teacher, base_model=teacher, decoder_model=teacher))
    fp32 = FP32Teacher(teacher_fp32)
    decoder = OriginalDecoder('cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    rows = []
    for record, diagnostic_row in zip(contract['scenes'], diagnostic['rows'], strict=True):
        if record['metadata']['split'] != 'train' or record['metadata']['seed'] in HOLDOUT_SEEDS:
            raise ValueError('TRAIN only.')
        path = Path(record['root'])/record['file']
        if file_sha(path) != record['capture_sha256']:
            raise ValueError('Diagnostic capture changed.')
        region_bf16, region_fp32 = DecodedRegion(path, 'cuda'), DecodedRegion(path, 'cuda')
        noise_seed = diagnostic_row['noise_seed']
        region_bf16.refresh_noise(teacher_bf16, weight, noise_seed)
        region_fp32.refresh_noise(fp32, weight, noise_seed)
        ref, actual = region_bf16.target, region_fp32.target
        low_ref, low_actual = low_height(ref[None]), low_height(actual[None])
        height_lf = blur(low_actual-low_ref, 8.)[..., 16:48, 16:48]
        latent_lf = blur((actual[:4]-ref[:4]).float()[:, None], 8.)[..., 16:48, 16:48]
        normalized_mse = latent_lf.square().mean((-3, -2, -1))/variance[:4]
        high_ref = reconstructed_height(decoder(ref[None], region_bf16.decoder_noise), ref[None])[0, 0, 64:-64, 64:-64]
        high_actual = reconstructed_height(decoder(actual[None], region_fp32.decoder_noise), actual[None])[0, 0, 64:-64, 64:-64]
        row = dict(index=diagnostic_row['index'], seed=record['metadata']['seed'], split='train',
            kind=record['case_kind'], coarse=record['coarse_variant'], noise_seed=noise_seed,
            height_lf_mse_m2=float(height_lf.square().mean()),
            c0_c3_lf_normalized_mse=[float(v) for v in normalized_mse],
            metrics={'3': height_stats(low_actual[0, 0, 16:48, 16:48], low_ref[0, 0, 16:48, 16:48]),
                     '0': height_stats(high_actual, high_ref)})
        rows.append(row)
        print(json.dumps(row), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['kind']].append(row)
    floors = {kind: dict(height_lf_rms_m=float(np.sqrt(np.mean([r['height_lf_mse_m2'] for r in group]))),
        c0_c3_lf_normalized_rms=np.sqrt(np.mean([r['c0_c3_lf_normalized_mse'] for r in group], axis=0)).tolist())
        for kind, group in grouped.items()}
    if any(f['height_lf_rms_m'] <= 0 or min(f['c0_c3_lf_normalized_rms']) <= 0 for f in floors.values()):
        raise ValueError('Nonzero observed precision floors required; no arbitrary epsilon calibration.')
    atomic_json(output, dict(status='complete', rows=rows, floors=floors, split='train', updates=0,
        diagnostic_sha256=file_sha(args.diagnostic/'report.json'),
        statistics_sha256=file_sha(stats_path), gpu=torch.cuda.get_device_name(),
        code_sha256={name: file_sha(REPO/'distill'/name) for name in
                     ('native_level_floor.py', 'native_level_diagnostic.py', 'native_decoded.py', 'native_fusion.py')},
        scope='Same fresh TRAIN noises and conditioning; base FP32 vs BF16, solver and decoder BF16.'))


if __name__ == '__main__':
    main()
