"""Independent checks for world64 attention lines and deep-level KD conflicts."""
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image,ImageDraw
import torch

from distill.common import DATA,atomic_json,atomic_write,external_path
from distill.field_base import FieldBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.field_loss import teacher_features,central_kd
from distill.native_base import teacher_source
from distill.native_progress_images import shade
from distill.wet_mountains import ROOT,digest
from distill.wet_mountain_progress import selected_validation
from terrain_diffusion.models.edm_unet import EDMUnet2D


def line_stats(candidate,reference,origin):
    h,w=reference.shape
    yy,xx=np.mgrid[:h,:w];dy=(yy+origin[0])%64;dx=(xx+origin[1])%64
    distance=np.minimum(np.minimum(dy,64-dy),np.minimum(dx,64-dx))
    land=reference>0;valid=land.copy();valid[:16]=valid[-16:]=False;valid[:,:16]=valid[:,-16:]=False
    online=valid & (distance<=2);offline=valid & (distance>=8) & (distance<=24)
    residual=candidate-reference
    tpi=(candidate-gaussian_filter(candidate,8))-(reference-gaussian_filter(reference,8))
    def rms(array,mask):return float(np.sqrt(np.mean(array[mask]**2))) if mask.any() else None
    a,b=rms(residual,online),rms(residual,offline)
    c,d=rms(tpi,online),rms(tpi,offline)
    return dict(line_pixels=int(online.sum()),off_line_pixels=int(offline.sum()),
        line_error_rms_m=a,off_line_error_rms_m=b,line_error_ratio=a/b if b and a is not None else None,
        line_tpi8_error_rms_m=c,off_line_tpi8_error_rms_m=d,
        line_tpi8_error_ratio=c/d if d and c is not None else None)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',default=str(DATA/'_base'))
    args=parser.parse_args();output=external_path(args.output)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=FieldBase(source['config']).cuda().eval().requires_grad_(False);model.load_state_dict(source['ema'])
    records,_=selected_validation(ROOT,'coarse4');rows=[];images=[]
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    for record in records:
        scene=FieldScene(ROOT/record['file'],'cuda')
        result=field_scene_forward(model,scene,features=True)
        q=result['core'][0,4]*38.6-31.4;p=scene.target[4]*38.6-31.4
        candidate=(q.sign()*q.square()).cpu().numpy();reference=(p.sign()*p.square()).cpu().numpy()
        origin=(scene.metadata['y'],scene.metadata['x'])
        line=line_stats(candidate,reference,origin)
        _,blocks=central_kd(result,teacher_features(teacher,scene))
        levels={}
        for name,value in blocks.items():
            phase,side,block=name.split(':')[0],name.split(':')[1].split('.')[0],name.split(':')[1].split('.')[1]
            key=phase+':'+side+'.'+block.split('_')[0]
            levels.setdefault(key,[]).append(float(value))
        rows.append(dict(seed=scene.metadata['seed'],lines=line,
            kd_levels={k:float(np.mean(v)) for k,v in levels.items()},
            outer16_mae_m=float(np.abs(candidate-reference)[np.pad(np.ones((96,96),bool),16)==0].mean()),
            inner64_mae_m=float(np.abs(candidate-reference)[32:96,32:96].mean())))
        panel=shade(candidate,240,1.5).resize((768,768),Image.Resampling.NEAREST)
        draw=ImageDraw.Draw(panel)
        for i in range(128):
            if (origin[0]+i)%64==0:draw.line((0,i*6,767,i*6),fill=(255,60,60),width=1)
            if (origin[1]+i)%64==0:draw.line((i*6,0,i*6,767),fill=(255,60,60),width=1)
        images.append(panel)
    stem=f'field-attention-lines-{source["step"]:06d}'
    canvas=Image.new('RGB',(768*len(images),768))
    for i,panel in enumerate(images):canvas.paste(panel,(i*768,0))
    atomic_write(output/f'{stem}.png',lambda f:canvas.save(f,format='PNG'))
    atomic_json(output/f'{stem}.json',dict(checkpoint_sha256=digest(args.checkpoint),step=source['step'],rows=rows,
        source_sha256=digest(__file__),accepted=False,note='Native attention world64 lines are red overlays. Quantitative error/TPI/KD data accompany this diagnostic; app seams are separate.'))
    (output/f'{stem}.txt').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(step=source['step'],rows=rows)),flush=True)


if __name__=='__main__':main()
