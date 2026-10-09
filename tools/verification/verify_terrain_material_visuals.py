"""Reproducible albedo/shaded views for material review, without neural jobs.

python verify_terrain_material_visuals.py [--before tmp/render_review_before.py]
Output: output/render-review/{scene}.png and comparison.png.
The optional before source is a snapshot of the material under review.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from terrain_render import surface_material,terrain_derivatives
from terrain_soil import coordinates,seed_value
from terrain_lighting import relief_intensity
from test_terrain_render import SOIL

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before',type=Path)
parser.add_argument('--output',type=Path,default=Path('output/render-review'))
parser.add_argument('--before-label',default='Claude · avant')
args=parser.parse_args()
old=None
if args.before:
    spec=importlib.util.spec_from_file_location('before_material',args.before)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
out=args.output;out.mkdir(parents=True,exist_ok=True)
n=384;bounds=(-20e6,-10e6,20e6,10e6)
scenes=[
    ('foothills',60,500,350,[25,350,.0065,6,420,.0065],.5),
    ('tropical',120,1300,1100,[30,900,.006,25,1000,.006],.5),
    ('mountain-valleys',60,2600,1500,[30,550,.006,8,550,.006],.5),
    ('arid-mountains',120,2800,1500,[38,60,.0055,24,80,.0055],.5),
    ('winter-temperate',60,850,500,[20,450,.006,-4,450,.006],0),
    ('cold-mountains',30,3700,1400,[20,650,.0065,-4,650,.0065],.5),
    ('distant-ecotone',2000,900,700,[24,180,.0065,4,220,.0065],.5),
    ('southern-summer',120,2200,1500,[7,500,.006,29,600,.006],0),
]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
rows=[]
for name,r,base,amplitude,climate,season in scenes:
    xs=(np.arange(n)+.5-n/2)*r+2e6;ys=(np.arange(n)+.5-n/2)*r-4e6
    y,x=np.meshgrid(ys+4e6,xs-2e6,indexing='ij')
    # A broad DEM must itself be band-limited; tiny synthetic ridges sampled
    # at 2 km would test terrain aliasing rather than material filtering.
    terrain_scale=max(1,r/120)
    x=x/terrain_scale;y=y/terrain_scale
    relief=(np.sin(x/3100+.3*np.cos(y/8200))+.65*np.cos(y/5700)+.13*np.sin(x/730)*np.cos(y/1230))
    dem=(base+amplitude*relief).astype(np.float32)
    g,tpi=terrain_derivatives(dem,r);point=coordinates(xs,ys,bounds)
    north=np.broadcast_to([0,-1],g.shape)
    soil=np.broadcast_to(SOIL[:,None,None],(9,n,n))
    seasons=np.broadcast_to(np.array(climate,np.float32)[:,None,None],(6,n,n))
    settings={'season':season}
    inputs=(dem,g,tpi,point,north,r,soil,seasons,seed_value(42),settings)
    rgb=surface_material(*inputs)
    light=relief_intensity(dem,r,{'strength':.6,'ambient':.45})
    current=Image.fromarray((np.clip(rgb*light[...,None],0,1)*255).astype(np.uint8))
    current.save(out/(name+'.png'))
    # Albedo is included so cast shading cannot conceal mask problems.
    images=[Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)),current]
    labels=['Albedo actuel','Rendu actuel']
    if old:
        before=old.surface_material(*inputs)
        images.insert(0,Image.fromarray((np.clip(before*light[...,None],0,1)*255).astype(np.uint8)))
        labels.insert(0,args.before_label)
    row=Image.new('RGB',(n*len(images),n+44),'#f4f4f0');d=ImageDraw.Draw(row)
    for i,(im,label) in enumerate(zip(images,labels)):
        row.paste(im,(i*n,44));d.text((i*n+10,8),f'{name} · {r} m · {label}',fill='#23302a',font=font)
    rows.append(row)
sheet=Image.new('RGB',(rows[0].width,sum(row.height for row in rows)))
offset=0
for row in rows:sheet.paste(row,(0,offset));offset+=row.height
sheet.save(out/'comparison.png')
print(out/'comparison.png')
