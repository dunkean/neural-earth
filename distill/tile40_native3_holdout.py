"""Decode reserved TEST once after freezing a VAL-selected native base."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.native_base import teacher_source
from distill.native_progress_images import shade
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.teacher import Worlds
from distill.tile40_conditioning import CLIMATES, RELIEFS, SketchConditioning
from distill.tile40_evaluate import compare_height
from distill.tile40_height import guided_reference, resample_tile
from distill.tile40_native3 import load_packed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'tile40-20261020/native3-reference-heldout-v1')
    parser.add_argument('--packed',type=Path,required=True,help='Omit for an uncompressed original repeat control.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=30)
    parser.add_argument('--freeze-proof',type=Path,required=True)
    args = parser.parse_args()
    proof=json.loads(args.freeze_proof.read_text())
    metadata=json.loads((args.packed/'manifest.json').read_text())
    if metadata['sha256']!=proof['model_sha256'] or args.packed.resolve()!=Path(proof['packed']).resolve():
        raise ValueError('Heldout candidate changed after freezing.')
    if file_sha(args.packed/'manifest.json')!=proof['packed_manifest_sha256']:
        raise ValueError('Heldout model metadata changed after freezing.')
    total_bytes=sum(p.stat().st_size for p in args.packed.rglob('*') if p.is_file())
    if total_bytes!=proof['model_bytes'] or total_bytes>=100000000:
        raise ValueError('Heldout model inventory or byte budget changed.')
    if file_sha(args.dataset/'manifest.json')!=proof['test_manifest_sha256']:
        raise ValueError('Heldout test manifest changed.')
    if args.limit<1:
        raise ValueError('Positive bounded validation count required.')
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve physical evaluations;use a fresh directory.')
    output.mkdir(parents=True)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = args.dataset/'manifest.json'
    manifest = json.loads(source.read_text())
    cases = sorted((r for r in manifest['cases'] if r['split']=='test'),
        key=lambda r:(r['seed']//10,list(CLIMATES).index(r['climate']),RELIEFS.index(r['relief'])))[:args.limit]
    worlds = Worlds(cases[0]['seed'])
    packed = None
    if args.packed:
        from distill.tile40_native3_lora import ARCHITECTURE as RESIDUAL_ARCHITECTURE,load_residual
        loader=load_residual if metadata['architecture']==RESIDUAL_ARCHITECTURE else load_packed
        base,packed = loader(args.packed,'cuda')
        worlds.loaded.base_model = base
    coarse,_ = load_student(REPO/'distill/weights/coarse2.pt','cuda')
    if file_sha(REPO/'distill/weights/coarse2.pt')!=manifest['internal_coarse_checkpoint']['sha256']:
        raise ValueError('Frozen coarse context differs.')
    rows = []
    for case in cases:
        path = args.dataset/case['file']
        if file_sha(path)!=json.loads(path.with_suffix('.json').read_text())['file_sha256']:
            raise ValueError('Frozen TEST changed.')
        with np.load(path,allow_pickle=False) as saved:
            sketch,reference = saved['sketch_m'].copy(),saved['height_m'].copy()
        world = worlds.create(case['seed'],'natural')
        from distill.inference import install
        install(world,dict(coarse=coarse))
        world.bind()
        world.synthetic_map_factory = SketchConditioning(sketch,case['climate'])
        began = time.monotonic()
        try:
            native = world.get(-256,-256,1600,1600,with_climate=False)['elev']
            if isinstance(native,torch.Tensor):
                native = native.float().cpu().numpy()
            height = resample_tile(guided_reference(native,sketch)['native_height'])
            torch.cuda.synchronize()
            seconds = time.monotonic()-began
            metrics = compare_height(height,reference,reference_cache=DATA/'cache/tile40-reference-metrics-v1')
            row = dict(case=case,metrics=metrics,timing=dict(native_total_seconds=seconds),
                paired_spatial_band_correlation=float(np.mean([b['correlation'] for b in metrics['bands']])))
            rows.append(row)
            atomic_write(output/case['file'],lambda handle:np.savez_compressed(handle,
                height_m=height,reference_height_m=reference,sketch_m=sketch))
            pair = Image.new('RGB',(1024,512))
            pair.paste(shade(reference,40000/1334,1.5).resize((512,512)),(0,0))
            pair.paste(shade(height,40000/1334,1.5).resize((512,512)),(512,0))
            atomic_write(output/Path(case['file']).with_suffix('.png'),lambda handle:pair.save(handle,format='PNG'))
            atomic_json(output/'report.json',dict(status='running',rows=rows))
            print(json.dumps(dict(case=case,height_mae_m=metrics['height_mae_m'],
                band_correlation=row['paired_spatial_band_correlation'],seconds=seconds)),flush=True)
        finally:
            world.close()
    report = dict(status='complete',rows=rows,architecture='original-native-two-stage',
        packed_base=None if args.packed is None else str(args.packed),
        base_sha256=file_sha(teacher_source()/'diffusion_pytorch_model.safetensors') if packed is None else packed['sha256'],
        packed_bytes=None if packed is None else sum(p.stat().st_size for p in args.packed.rglob('*') if p.is_file()),
        step=0 if packed is None else packed['step'],selection='original' if packed is None else packed['selection'],
        original_repeat_control=packed is None,decoder='original',
        decoder_sha256=file_sha(teacher_source().parent/'decoder_model/diffusion_pytorch_model.safetensors'),
        coarse_sha256=file_sha(REPO/'distill/weights/coarse2.pt'),dataset_manifest_sha256=file_sha(source),
        producer_sha256=file_sha(Path(__file__)),quality_accepted=False,heldout_test=True,freeze_proof_sha256=file_sha(args.freeze_proof),
        scope='Full native two-stage generation,original decoder,paired noise;only base is capped100MB;frozen TEST;no training or tuning on TEST.')
    atomic_json(output/'report.json',report)


if __name__=='__main__':main()
