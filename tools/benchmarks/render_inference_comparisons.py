"""CPU-only galleries of physical benchmark NPZ terrain arrays, never AI imagery."""
from __future__ import annotations

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()


import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LightSource
import numpy as np


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            result.update(block)
    return result.hexdigest()


def load(path, allow_running=False):
    path=Path(path).resolve()
    receipt_path=path.with_suffix('.json')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
    status=receipt.get('status',receipt.get('state','unknown'))
    if status=='running' and not allow_running:
        raise ValueError(f'{path.name}: benchmark running; wait for completion')
    before=path.stat()
    with np.load(path,allow_pickle=False) as archive:
        arrays={k:archive[k].astype(np.float64) for k in archive.files if k.endswith('_elev')}
    if not arrays:
        raise ValueError(f'No *_elev arrays: {path}')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError(f'File changed during read: {path}')
    if any(x.ndim!=2 or min(x.shape)<3 for x in arrays.values()):
        raise ValueError('Expected 2D elevation arrays with at least 3 pixels per side')
    return dict(path=str(path),sha256=digest(path),label=receipt.get('label',path.stem),
                status=status,receipt_sha=digest(receipt_path) if receipt else None,
                samples={s['key']:s for s in receipt.get('samples',[])},arrays=arrays)


def metrics(reference,candidate):
    finite=np.isfinite(reference)&np.isfinite(candidate)
    if not finite.any():
        raise ValueError('No comparable finite pixels')
    error=candidate[finite]-reference[finite]
    return dict(mae_m=float(np.mean(np.abs(error))),rmse_m=float(np.sqrt(np.mean(error**2))),
                max_abs_m=float(np.abs(error).max()),bias_m=float(error.mean()),
                nonfinite_pairs=int((~finite).sum()),numeric_exact=bool(np.array_equal(reference,candidate)))


def slope(elevation,resolution):
    dy,dx=np.gradient(elevation,resolution)
    return np.degrees(np.arctan(np.hypot(dx,dy)))


def crop_at(shape,position,width=64):
    h,w=shape
    ch,cw=min(width,h),min(width,w)
    row=max(0,min(int(position[0])-ch//2,h-ch))
    col=max(0,min(int(position[1])-cw//2,w-cw))
    return (slice(row,row+ch),slice(col,col+cw))


def terrain_rgb(elevation,norm,cmap,resolution):
    values=np.nan_to_num(elevation,nan=0.,posinf=0.,neginf=0.)
    rgb=cmap(norm(values))[...,:3]
    light=LightSource(azdeg=315,altdeg=45)
    # Illumination uses real metric spacing and no vertical exaggeration.
    dy,dx=np.gradient(values,-resolution,resolution)
    normals=np.stack((-dx,-dy,np.ones_like(values)),axis=-1)
    normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
    # Do not independently stretch each image's shade range (hillshade default).
    shade=np.clip(normals@light.direction,0,1)
    result=light.blend_soft_light(rgb,shade[...,None])
    result[~np.isfinite(elevation)]=[1,0,1]
    return result


def decorate(axis,title,shape,resolution):
    axis.set_title(title,fontsize=10)
    axis.set_xlabel('Distance locale est (km)')
    axis.set_ylabel('Distance locale sud (km)')


def render_case(key,ref,cand,metadata,output,norm,cmap,error_limit,slope_limit):
    match=re.search(r'lod(-?\d+)',key)
    if not match:
        raise ValueError(f'LOD absent from key: {key}')
    lod=int(match.group(1))
    resolution=30.*2**lod
    if ref.shape!=cand.shape:
        raise ValueError(f'Shape mismatch: {key}')
    stats=metrics(ref,cand)
    error=cand-ref
    rs,cs=slope(ref,resolution),slope(cand,resolution)
    extent=(0,ref.shape[1]*resolution/1000,ref.shape[0]*resolution/1000,0)
    rrgb,crgb=terrain_rgb(ref,norm,cmap,resolution),terrain_rgb(cand,norm,cmap,resolution)
    title=f'{key} · LOD {lod} · {resolution:g} m/pixel'
    if metadata:
        title+=f' · seed {metadata.get("seed","?")} · tuile ({metadata.get("tx","?")}, {metadata.get("ty","?")})'
    fig,axes=plt.subplots(2,3,figsize=(15,9.5),constrained_layout=True)
    metric_text=f'MAE {stats["mae_m"]:.6g} m · RMSE {stats["rmse_m"]:.6g} m · max |écart| {stats["max_abs_m"]:.6g} m'
    fig.suptitle(title+'\nRéférence à gauche ; candidat au centre ; écarts à droite\n'+metric_text,fontsize=13)
    for col,(rgb,label) in enumerate(((rrgb,'Référence — altitude et ombrage'),(crgb,'Candidat — altitude et ombrage'))):
        axes[0,col].imshow(rgb,extent=extent,interpolation='nearest')
        decorate(axes[0,col],label,ref.shape,resolution)
    height_bar=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    fig.colorbar(height_bar,ax=axes[0,:2],shrink=.7,label='Altitude (m), échelle commune à toute la galerie')
    image=axes[0,2].imshow(error,cmap='RdBu_r',vmin=-error_limit,vmax=error_limit,
                           extent=extent,interpolation='nearest')
    decorate(axes[0,2],'Écart signé : candidat − référence',ref.shape,resolution)
    fig.colorbar(image,ax=axes[0,2],shrink=.75,label='Écart d’altitude (m)')
    for col,(data,label) in enumerate(((rs,'Pente référence'),(cs,'Pente candidat'))):
        image=axes[1,col].imshow(data,cmap='magma',vmin=0,vmax=slope_limit,extent=extent,interpolation='nearest')
        decorate(axes[1,col],label,ref.shape,resolution)
    fig.colorbar(image,ax=axes[1,:2],shrink=.7,label='Pente (degrés), différences finies au pas physique')
    delta=cs-rs
    dlimit=max(float(np.nanmax(np.abs(delta))),.001)
    image=axes[1,2].imshow(delta,cmap='PuOr',vmin=-dlimit,vmax=dlimit,extent=extent,interpolation='nearest')
    decorate(axes[1,2],'Variation de pente',ref.shape,resolution)
    fig.colorbar(image,ax=axes[1,2],shrink=.75,label='Écart de pente (°) — échelle propre au cas')
    overview=output/f'{key}-overview.png'
    fig.savefig(overview,dpi=130)
    plt.close(fig)

    steep=np.unravel_index(np.nanargmax(rs),rs.shape)
    worst=np.unravel_index(np.nanargmax(np.abs(error)),error.shape) if stats['max_abs_m'] else (ref.shape[0]//2,ref.shape[1]//2)
    crops=[('Relief : pente maximale de la référence',crop_at(ref.shape,steep)),
           ('Écart maximal' if stats['max_abs_m'] else 'Centre — écarts tous nuls',crop_at(ref.shape,worst))]
    fig,axes=plt.subplots(2,3,figsize=(15,9.5),constrained_layout=True)
    fig.suptitle(title+'\nZooms sans interpolation — même illumination et mêmes échelles',fontsize=14)
    for row,(label,crop) in enumerate(crops):
        y,x=crop
        subextent=(x.start*resolution/1000,x.stop*resolution/1000,y.stop*resolution/1000,y.start*resolution/1000)
        for col,(rgb,name) in enumerate(((rrgb,'Référence'),(crgb,'Candidat'))):
            axes[row,col].imshow(rgb[crop],extent=subextent,interpolation='nearest')
            axes[row,col].set_title(f'{label}\n{name}',fontsize=10)
        image=axes[row,2].imshow(error[crop],cmap='RdBu_r',vmin=-error_limit,vmax=error_limit,
                                extent=subextent,interpolation='nearest')
        axes[row,2].set_title('Écart signé (m)')
        fig.colorbar(image,ax=axes[row,2],shrink=.75,label='m')
        for axis in axes[row]:
            axis.set_xlabel('Distance locale est (km)')
            axis.set_ylabel('Distance locale sud (km)')
    zoom=output/f'{key}-zooms.png'
    fig.savefig(zoom,dpi=130)
    plt.close(fig)

    # A diagnostic cross-section, not evidence of neighboring-tile continuity.
    fig,axes=plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
    fig.suptitle(title+'\nProfils locaux centre / bord intérieur — aucune preuve de raccord global')
    lines=[('Ligne centrale',ref.shape[0]//2),('Bord intérieur (halo24 si disponible)',min(24,ref.shape[0]-2))]
    x=np.arange(ref.shape[1])*resolution/1000
    for row,(label,index) in enumerate(lines):
        axes[row,0].plot(x,ref[index],label='Référence',color='#12616b')
        axes[row,0].plot(x,cand[index],label='Candidat',color='#e98f2e',linestyle='--')
        axes[row,0].set_title(f'{label}, y={index*resolution/1000:g} km')
        axes[row,0].set_ylabel('Altitude (m)')
        axes[row,0].legend()
        axes[row,1].plot(x,error[index],color='#a31e44')
        axes[row,1].set_ylim(-error_limit,error_limit)
        axes[row,1].set_title('Écart candidat − référence')
        axes[row,1].set_ylabel('Écart (m)')
        for ax in axes[row]:
            ax.set_xlabel('Distance locale est (km)')
            ax.grid(alpha=.2)
    profiles=output/f'{key}-profiles.png'
    fig.savefig(profiles,dpi=130)
    plt.close(fig)
    return dict(key=key,lod=lod,resolution_m=resolution,shape=list(ref.shape),metrics=stats,
                metadata={k:metadata[k] for k in ('seed','tx','ty','stage') if k in metadata},
                images=[p.name for p in (overview,zoom,profiles)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-running',action='store_true',help='Exploratory snapshot only; file must remain stable while read')
    args=parser.parse_args()
    reference=load(args.reference,args.allow_running)
    candidates=[load(path,args.allow_running) for path in args.candidate]
    finite=np.concatenate([x[np.isfinite(x)] for x in reference['arrays'].values()])
    lo,hi=min(-1.,float(finite.min())),max(1.,float(finite.max()))
    norm=TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi)
    cmap=LinearSegmentedColormap.from_list('physical_terrain',
        [(0,'#122c57'),(.45,'#66b8c6'),(.4999,'#d6f0ee'),(.5,'#467549'),(.68,'#a6a562'),(.85,'#a8876e'),(1,'#fcf9f0')])
    args.output.mkdir(parents=True,exist_ok=True)
    pages=[]
    report=dict(reference={k:v for k,v in reference.items() if k not in ('arrays','samples')},
                altitude_scale_m=[lo,hi],candidates=[],cpu_only=True,
                renderer_sha256=digest(__file__),
                warning='Local arrays including benchmark halo; not proof of global tile seams. No screenshot or GPU timing.')
    for ordinal,candidate in enumerate(candidates):
        keys=sorted(set(reference['arrays'])&set(candidate['arrays']))
        if not keys:
            raise ValueError(f'No matching elevation keys: {candidate["path"]}')
        safe=re.sub('[^a-zA-Z0-9_-]','_',candidate['label'])
        folder=args.output/f'{ordinal:02d}-{safe}'
        folder.mkdir(exist_ok=True)
        for key in keys:
            if reference['arrays'][key].shape!=candidate['arrays'][key].shape:
                raise ValueError(f'Shape mismatch: {key}')
            rmeta=reference['samples'].get(key.removesuffix('_elev'),{})
            cmeta=candidate['samples'].get(key.removesuffix('_elev'),{})
            for field in ('seed','lod','tx','ty'):
                if field in rmeta and field in cmeta and rmeta[field]!=cmeta[field]:
                    raise ValueError(f'Physical footprint mismatch {key}/{field}')
        error_limit=max(1.,max(float(np.nanmax(np.abs(candidate['arrays'][k]-reference['arrays'][k]))) for k in keys))
        slope_limit=max(1.,max(float(np.nanmax(slope(data[k],30.*2**int(re.search(r'lod(-?\d+)',k).group(1)))))
                              for k in keys for data in (reference['arrays'],candidate['arrays'])))
        cases=[]
        for key in keys:
            cases.append(render_case(key.removesuffix('_elev'),reference['arrays'][key],candidate['arrays'][key],
                reference['samples'].get(key.removesuffix('_elev'),{}),folder,norm,cmap,error_limit,slope_limit))
        summary={k:v for k,v in candidate.items() if k not in ('arrays','samples')}
        summary.update(cases=cases,error_scale_m=[-error_limit,error_limit],
                       missing_reference_keys=sorted(set(reference['arrays'])-set(candidate['arrays'])))
        summary['exceeds_height_gate_1m']=any(case['metrics']['max_abs_m']>1 for case in cases)
        report['candidates'].append(summary)
        status=html.escape(candidate['status'])
        pages.append(f'<h2>{html.escape(candidate["label"])} <small>reçu : {status}</small></h2>')
        if summary['exceeds_height_gate_1m']:
            pages.append('<p class="warning"><b>REJETÉ PAR LE SEUIL PHYSIQUE DE 1 MÈTRE.</b> Cette variante est présentée uniquement pour examiner ses artefacts ; elle ne doit pas être activée.</p>')
        pages.append('<p>Un reçu « failed » est une expérience rejetée, affichée pour diagnostic. Les mesures ci-dessous portent uniquement sur les arrays disponibles.</p>')
        if summary['missing_reference_keys']:
            pages.append('<p class="warning">Corpus incomplet : '+html.escape(', '.join(summary['missing_reference_keys']))+'</p>')
        for case in cases:
            stats=case['metrics']
            pages.append(f'<section><h3>{html.escape(case["key"])} · {case["resolution_m"]:g} m/pixel</h3>'
                f'<p>MAE {stats["mae_m"]:.6g} m · RMSE {stats["rmse_m"]:.6g} m · max |écart| {stats["max_abs_m"]:.6g} m'
                f' · biais {stats["bias_m"]:.6g} m · pixels non finis {stats["nonfinite_pairs"]}</p>')
            for image,label in zip(case['images'],('Vue générale et pentes','Zooms locaux','Profils locaux')):
                url=f'{folder.name}/{image}'
                pages.append(f'<details {"open" if label=="Vue générale et pentes" else ""}><summary>{label}</summary>'
                    f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}" alt="{html.escape(label)}"></a></details>')
            pages.append('</section>')
    document='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparaisons du moteur terrain</title><style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:0 20px;background:#f2f4f2;color:#20342f}h1,h2,h3{line-height:1.2}h2{margin-top:50px}small{font-size:15px}section{background:white;margin:22px 0;padding:20px;border-radius:10px}img{width:100%;height:auto}summary{cursor:pointer;padding:12px;background:#e3eae5}.warning{color:#942e31}p{max-width:1050px;line-height:1.5}a{color:#12616b}</style>
<h1>Comparaisons du moteur d’inférence terrain</h1>
<p>Images calculées directement depuis les altitudes du benchmark. Référence et candidat partagent l’échelle hypsométrique et l’éclairage (azimut315°, hauteur45°, sans exagération verticale). LOD3 : <b>240m/pixel</b> ; LOD2 : <b>120m/pixel</b>. Cliquer une image pour l’ouvrir à sa taille complète.</p>
<p>Écarts signés en mètres : rouge = candidat plus haut, bleu = plus bas. L’échelle d’écart est commune aux cas d’un candidat, avec minimum±1m même si tous les écarts sont nuls. Les gradients suivent le pas physique. Les tracés au bord d’un crop ne certifient pas les raccords entre tuiles : ce corpus ne contient pas nécessairement les voisins.</p>
<p>Un aperçu latent n’est pas le DEM final. Ces vues ne constituent ni un rendu WebGPU, ni une validation géographique, ni une mesure de performance. La légende d’altitude est fixée par la référence ; les valeurs candidates hors plage prennent les couleurs extrêmes.</p>'''
    document+=''.join(pages)+'<p><a href="comparison.json">Provenance, SHA et métriques JSON</a></p></html>'
    (args.output/'index.html').write_text(document,encoding='utf-8')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(gallery=str((args.output/'index.html').resolve()),candidates=len(candidates),cases=sum(len(x['cases']) for x in report['candidates']))))


if __name__=='__main__':
    main()
