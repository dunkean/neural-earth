"""Matched autonomous native QAT terrain, including wet incised mountains."""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
import torch

from distill.common import DATA,atomic_json,atomic_write,external_path
from distill.native_fp8 import load_fp8
from distill.native_decoded import DecodedRegion,OriginalDecoder,decoded_region_forward
from distill.native_fusion import window_outputs
from distill.native_progress_images import shade,LABELS
from distill.decoded_loss import reconstructed_height
from distill.terrain_shape import compare_shape
from distill.wet_mountains import ROOT,KIND,digest
from distill.wet_mountain_progress import selected_validation


@torch.no_grad()
def evaluate(checkpoint,output,label='qat',coarse='coarse4',quantized=True):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['architecture']!='native-fp8':raise ValueError('Native precision checkpoint required.')
    model=load_fp8(saved,'cuda');model.set_quantized(quantized)
    root=DATA/'crops/native-mixed-scenes-v2';manifest=json.loads((root/'manifest.json').read_text())
    selected=[]
    for kind in LABELS:
        rows=[r for r in manifest['scenes'] if r['metadata']['split']=='val' and
              r['case_kind']==kind and r['coarse_variant']==coarse]
        selected.append((root,sorted(rows,key=lambda r:(r['metadata']['seed'],r['file']))[0]))
    selected += [(ROOT,r) for r in selected_validation(ROOT,coarse)[0]]
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);decoder=OriginalDecoder('cuda')
    variance=torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'],device='cuda')
    fields=[];rows=[]
    for source,record in selected:
        region=DecodedRegion(source/record['file'],'cuda');result=decoded_region_forward(model,region,weight)
        first=region.stages[0]
        actual=window_outputs(first['inputs'],result['velocity0'],first['labels'],region.dtype)
        original=window_outputs(first['inputs'],first['target'],first['labels'],region.dtype)
        channel0_mse=float((actual[:,0]-original[:,0]).square().mean()/variance[0])
        heights=[]
        for latent in (region.target,result['core']):
            q=latent[4].float()*38.6-31.4
            fine=reconstructed_height(decoder(latent[None],region.decoder_noise),latent[None])[0,0].cpu().numpy()[64:-64,64:-64]
            heights.append({3:(q.sign()*q.square()).cpu().numpy(),0:fine})
        for lod in (3,0):
            reference,candidate=heights[0][lod],heights[1][lod];shape=compare_shape(candidate,reference)
            tpi=next(r for r in shape['topographic_position'] if r['sigma_pixels']==8.)
            rows.append(dict(step=saved['step'],kind=record['case_kind'],seed=record['metadata']['seed'],coarse=coarse,lod=lod,
                mae_m=float(np.abs(candidate-reference).mean()),correlation=tpi.get('tpi_correlation'),
                recall=tpi.get('valley_retention'),rms=tpi.get('tpi_rms_ratio'),channel0_mse=channel0_mse,shape=shape))
            fields.append((reference,candidate))
    size,left,gap,header,row_h=512,245,12,125,556
    canvas=Image.new('RGB',(left+2*(size+gap),header+len(rows)*row_h),(24,24,24))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    names=LABELS|{KIND:'Haute montagne humide'}
    draw.text((12,10),f'Base natif192x3,64/32,T2 | {label} | etape {saved["step"]} | {coarse}',fill='white',font=font)
    draw.text((12,40),'Meme scene/bruit, decoder original. FP8 simule en calcul BF16 : aucune acceleration revendiquee.',fill='white',font=font)
    draw.text((left,83),'Teacher original BF16',fill='white',font=font)
    candidate_label=('FP8 simule non entraine' if saved.get('evaluation_only') else 'FP8 simule entraine') if quantized else 'Controle natif BF16'
    draw.text((left+size+gap,83),candidate_label,fill='white',font=font)
    for i,(row,pair) in enumerate(zip(rows,fields)):
        y=header+i*row_h
        draw.multiline_text((12,y+140),f'{names[row["kind"]]}\nLOD {row["lod"]}\nSeed {row["seed"]}\nErreur {row["mae_m"]:.2f} m\nRelief x1.5',fill='white',font=font,spacing=8)
        for col,elevation in enumerate(pair):
            canvas.paste(shade(elevation,30*2**row['lod'],1.5).resize((size,size),Image.Resampling.NEAREST),(left+col*(size+gap),y))
    stem=f'native-fp8-{label}-{coarse}-{saved["step"]:06d}'
    metadata=PngImagePlugin.PngInfo();metadata.add_text('CheckpointSHA256',digest(checkpoint))
    metadata.add_text('Description','Real native two-stage comparison. Static TRAIN scales, original decoder. CSV/JSON/text alternative. No acceptance or acceleration claim.')
    atomic_write(output/f'{stem}.png',lambda f:canvas.save(f,format='PNG',pnginfo=metadata))
    flat=[{k:v for k,v in r.items() if k!='shape'} for r in rows]
    with (output/f'{stem}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (output/f'{stem}.txt').write_text('Autonomous native64/32,T2 chain, original decoder. Real paired relief without smoothing. Metrics in CSV/JSON. FP8 simulation, BF16 computation.\n')
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),step=saved['step'],rows=rows,
                quantized=quantized,source_sha256=digest(__file__),gpu=torch.cuda.get_device_name(),accepted=False)
    atomic_json(output/f'{stem}.json',report)
    del model,decoder;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(dict(image=str(output/f'{stem}.png'),step=saved['step'],quantized=quantized)),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=DATA/'_base')
    parser.add_argument('--label',default='qat')
    parser.add_argument('--coarse',default='coarse4',choices=('original','coarse4','coarse2'))
    parser.add_argument('--no-quantization',action='store_true')
    args=parser.parse_args();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    evaluate(args.checkpoint,external_path(args.output),args.label,args.coarse,not args.no_quantization)


if __name__=='__main__':main()
