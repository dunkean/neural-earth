"""Real matched field-pupil terrain and shifted-crop continuity diagnostics."""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
import torch

from distill.common import DATA,REPO,atomic_json,atomic_write,external_path
from distill.field_base import FieldBase
from distill.field_phase import FieldPhaseBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.native_basis_training import inventory
from distill.native_decoded import OriginalDecoder
from distill.decoded_loss import reconstructed_height
from distill.native_progress_images import shade,LABELS
from distill.terrain_shape import compare_shape
from distill.wet_mountains import KIND,ROOT,digest
from distill.wet_mountain_progress import selected_validation


@torch.no_grad()
def evaluate(checkpoint,output,label,coarse='coarse4',seams=False):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    classes={'native-field':FieldBase,'native-field-phase':FieldPhaseBase}
    if saved['architecture'] not in classes:raise ValueError('Field checkpoint required.')
    model=classes[saved['architecture']](saved['config']).cuda().eval().requires_grad_(False);model.load_state_dict(saved['ema'])
    main=DATA/'crops/native-mixed-scenes-v2'
    manifest=json.loads((main/'manifest.json').read_text());selected=[]
    for kind in LABELS:
        pool=[r for r in manifest['scenes'] if r['metadata']['split']=='val' and
              r['case_kind']==kind and r['coarse_variant']==coarse]
        selected.append((main,sorted(pool,key=lambda r:(r['metadata']['seed'],r['file']))[0]))
    mountains,_=selected_validation(ROOT,coarse)
    selected += [(ROOT,r) for r in mountains]
    decoder=OriginalDecoder('cuda');rows=[];fields=[];seam_rows=[]
    from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
    for root,record in selected:
        scene=FieldScene(root/record['file'],'cuda')
        result=field_scene_forward(model,scene)
        initial_heights=[]
        for latent in (scene.initial,result['initial192'][0]):
            q=latent[4].float()*38.6-31.4
            initial_heights.append((q.sign()*q.square()).cpu().numpy()[32:-32,32:-32])
        initial_shape=next(r for r in compare_shape(initial_heights[1],initial_heights[0])['topographic_position'] if r['sigma_pixels']==8.)
        yy,xx=tuple(((scene.metadata[k]+95)//96)*96 for k in ('y','x'))
        oy,ox=yy-scene.metadata['y'],xx-scene.metadata['x']
        decoder_noise=torch.from_numpy(gaussian_noise_patch(scene.metadata['seed']+5819,
            yy*8,xx*8,512,512,channels=1,tile_h=512,tile_w=512))[None].cuda()
        heights=[]
        for latent in (scene.target[:,oy:oy+64,ox:ox+64],result['core'][0,:,oy:oy+64,ox:ox+64]):
            q=latent[4].float()*38.6-31.4
            fine=reconstructed_height(decoder(latent[None],decoder_noise),latent[None])[0,0].cpu().numpy()[64:-64,64:-64]
            heights.append({3:(q.sign()*q.square()).cpu().numpy(),0:fine})
        for lod in (3,0):
            reference,candidate=heights[0][lod],heights[1][lod]
            metrics=compare_shape(candidate,reference)
            tpi=next(r for r in metrics['topographic_position'] if r['sigma_pixels']==8.)
            rows.append(dict(step=saved['step'],kind=record['case_kind'],seed=scene.metadata['seed'],
                coarse=coarse,lod=lod,mae_m=float(np.abs(reference-candidate).mean()),
                correlation=tpi.get('tpi_correlation'),recall=tpi.get('valley_retention'),
                rms=tpi.get('tpi_rms_ratio'),shape=metrics,
                initial_correlation=initial_shape.get('tpi_correlation'),initial_rms=initial_shape.get('tpi_rms_ratio'),
                initial_mae_m=float(np.abs(initial_heights[1]-initial_heights[0]).mean())))
            fields.append((reference,candidate))
        if seams:
            for shift in ((0,32),(32,0)):
                moved=FieldScene(root/record['file'],'cuda',shift=shift)
                shifted=field_scene_forward(model,moved)['core']
                a,b=(result['core'][:,:,:,32:],shifted[:,:,:,:96]) if shift[1] else (result['core'][:,:,32:,:],shifted[:,:,:96,:])
                error=(a-b).float()
                qa,qb=a[:,4:5]*38.6-31.4,b[:,4:5]*38.6-31.4
                difference=(qa.sign()*qa.square()-qb.sign()*qb.square()).abs()
                seam_rows.append(dict(kind=record['case_kind'],seed=scene.metadata['seed'],axis='x' if shift[1] else 'y',
                    latent_mse=float(error.square().mean()),native_height_mae_m=float(difference.mean()),
                    native_height_max_m=float(difference.max()),
                    note='Same-world shifted input256 crops, shared128-core overlap. This is an early field-consistency diagnostic, not final application cold tile seams.'))
    size,left,gap,header,row_h=512,245,12,125,556
    canvas=Image.new('RGB',(left+2*(size+gap),header+len(rows)*row_h),(24,24,24))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    names=LABELS|{KIND:'Haute montagne humide'}
    draw.text((12,10),f'Eleve de champ, 2 phases,192x3 | {label} | etape {saved["step"]} | {coarse}',fill='white',font=font)
    draw.text((12,40),'Teacher compare sur la meme scene/bruit. Decoder original, sans lissage. Validation interne.',fill='white',font=font)
    draw.text((left,83),'Teacher original T2',fill='white',font=font)
    draw.text((left+size+gap,83),'Champ non entraine' if saved.get('evaluation_only') else 'Eleve entraine',fill='white',font=font)
    for i,(row,pair) in enumerate(zip(rows,fields)):
        y=header+i*row_h
        draw.multiline_text((12,y+140),f'{names[row["kind"]]}\nLOD {row["lod"]}\nSeed {row["seed"]}\nErreur {row["mae_m"]:.2f} m\nRelief x1.5',fill='white',font=font,spacing=8)
        for col,elevation in enumerate(pair):
            panel=shade(elevation,30*2**row['lod'],1.5).resize((size,size),Image.Resampling.NEAREST)
            canvas.paste(panel,(left+col*(size+gap),y))
    stem=f'field-{label}-{saved["step"]:06d}'
    metadata=PngImagePlugin.PngInfo();metadata.add_text('CheckpointSHA256',digest(checkpoint))
    metadata.add_text('Description','Real field comparison,7independent VAL scenes,14views; CSV and JSON provide metrics. No final acceptance.')
    atomic_write(output/f'{stem}.png',lambda f:canvas.save(f,format='PNG',pnginfo=metadata))
    with (output/f'{stem}.csv').open('w',newline='') as f:
        flat=[{k:v for k,v in r.items() if k!='shape'} for r in rows]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (output/f'{stem}.txt').write_text('Same-site real teacher and field pupil. Fixed lighting, no smoothing. Metrics in CSV/JSON.\n')
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),step=saved['step'],
        rows=rows,shifted_field_consistency=seam_rows,gpu=torch.cuda.get_device_name(),
        sources_sha256={p:digest(REPO/'distill'/p) for p in ('field_base.py','field_scene.py','field_evaluate.py')},accepted=False)
    atomic_json(output/f'{stem}.json',report)
    print(json.dumps(dict(image=str(output/f'{stem}.png'),step=saved['step'],seams=len(seam_rows))),flush=True)
    del model,decoder;gc.collect();torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=DATA/'_base')
    parser.add_argument('--label',default='initial')
    parser.add_argument('--coarse',default='coarse4',choices=('original','coarse4','coarse2'))
    parser.add_argument('--seams',action='store_true')
    args=parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label):
        parser.error('Safe strategy label required.')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    evaluate(args.checkpoint,external_path(args.output),args.label,args.coarse,args.seams)


if __name__=='__main__':main()
