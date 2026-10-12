"""TRAIN-only causal out_gain swap, with matched LOD3/0 footprints."""
import argparse
import json
from pathlib import Path

import torch

from distill.common import DATA,atomic_json,external_path
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion,OriginalDecoder,decoded_region_forward
from distill.decoded_loss import reconstructed_height
from distill.native_field_auxiliary import low_height
from distill.native_weight_quant import load_weight_quant,file_sha
from distill.wet_mountains import KIND


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--gain-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve previous causal evidence.')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    gain_source=torch.load(args.gain_source,map_location='cpu',weights_only=True,mmap=True)
    original=source['ema']['net.out_gain'].float().clone()
    swap=gain_source['ema']['net.out_gain'].float().clone()
    model=load_weight_quant(source,'cuda')
    records,_,_=inventory();selected=[]
    for kind in ('desert-plain','temperate-plain','wet-complex-coast','low-plain-sea',KIND):
        candidates=[r for r in records if r['case_kind']==kind and r['coarse_variant']=='coarse4']
        selected+=sorted(candidates,key=lambda r:(r['metadata']['seed'],r['file']))[:3 if kind==KIND else 1]
    decoder=OriginalDecoder('cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);rows=[]
    for record in selected:
        if record['metadata']['split']!='train':raise ValueError('TRAIN only.')
        region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
        target_low=low_height(region.target[None])[0,0,16:48,16:48]
        target_high=reconstructed_height(decoder(region.target[None],region.decoder_noise),region.target[None])[0,0,64:-64,64:-64]
        if target_high.shape!=(256,256):raise ValueError('Matched physical core changed.')
        variants={}
        for name,gain in [('baseline',original),('gain_swap',swap)]:
            model.net.out_gain.copy_(gain.to(model.net.out_gain.device))
            with model.cached_weights():core=decoded_region_forward(model,region,weight)['core']
            low=low_height(core[None])[0,0,16:48,16:48]
            high=reconstructed_height(decoder(core[None],region.decoder_noise),core[None])[0,0,64:-64,64:-64]
            variants[name]={}
            for lod,actual,target in [(3,low,target_low),(0,high,target_high)]:
                delta=actual.float()-target.float()
                variants[name][str(lod)]=dict(bias_m=float(delta.mean()),mae_m=float(delta.abs().mean()),
                    centered_mae_m=float((delta-delta.mean()).abs().mean()))
        row=dict(kind=record['case_kind'],seed=record['metadata']['seed'],split='train',file=record['file'],variants=variants)
        rows.append(row);print(json.dumps(row),flush=True)
    model.net.out_gain.copy_(original.to(model.net.out_gain.device))
    atomic_json(output,dict(status='complete',rows=rows,original_gain=float(original),swapped_gain=float(swap),
        source_sha256=file_sha(args.source),gain_source_sha256=file_sha(args.gain_source),
        weight_updates=0,only_out_gain_changed=True,gpu=torch.cuda.get_device_name(),
        footprint='Central32x32 latent /256x256 decoded physical samples; same area for both LODs.',
        scope='Causal scalar inference ablation on TRAIN; not a calibrated or accepted model.'))


if __name__=='__main__':main()
