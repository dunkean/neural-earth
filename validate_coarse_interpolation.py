"""LOD4 interpolation comparison from ONE learned coarse crop, then CPU only.

--capture requires exclusive GPU access. --source replays an existing crop on
CPU without importing terrain_app/server or initializing CUDA. --self-test uses
an explicitly synthetic field and produces no real-mountain evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT=Path(__file__).resolve().parent
RUNTIME=Path('E:/TerrainDiffusionRuntime/inference-engine-20261008/lod4-interpolation')
IMAGES=ROOT/'docs/performance_x5/lod4-images'
TILE,HALO,LOD,STEP,NATIVE,COARSE_STRIDE=256,24,4,16,30,256


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value):
    """Serialize NumPy values in both success and failure receipts."""
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported receipt value: {type(value).__name__}')


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=json_scalar),encoding='utf-8')


def sample_function(field,row0,col0):
    """Execute the CURRENT server function against a sealed CPU crop provider.

    Extract just this function's AST to avoid terrain_server's CUDA/server
    initialization. Its interpolation/formula code is not duplicated here.
    """
    path=ROOT/'terrain_server.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_field')
    requests=[]
    def block(world,source,x0,y0,x1,y1):
        if source!='coarse':
            raise ValueError('Only the sealed coarse crop is available')
        if not (col0<=x0<x1<=col0+field.shape[1] and row0<=y0<y1<=row0+field.shape[0]):
            raise ValueError('Requested halo outside saved crop; enlarge capture crop')
        requests.append([source,x0,y0,x1,y1])
        return field[y0-row0:y1-row0,x0-col0:x1-col0]
    namespace=dict(np=np,math=math,map_coordinates=map_coordinates,_field_block=block)
    isolated=ast.Module(body=[node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated),str(path),'exec'),namespace)
    return namespace['sample_field'],requests


def capture(args,output):
    import torch
    from terrain_app import load_pipeline,gpu_calls
    from terrain_inference import configure_world,inference_status
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    from terrain_window_scheduler import read_rect
    with torch.inference_mode():
        loaded=load_pipeline(args.seed)
        config={k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
        world=WorldPipeline(**(config|dict(seed=args.seed,latents_batch_size=16,dtype='bf16',
            cache_limit=512*1024**2,torch_compile=False,log_mode='silent')))
        try:
            world.coarse_model,world.base_model,world.decoder_model=(loaded.coarse_model,loaded.base_model,loaded.decoder_model)
            configure_world(world,world_profile=args.profile)
            if not world._terrain_profile.exact_kernels:
                raise ValueError('Capture requires the default exact-kernel runtime')
            world.bind()
            before=dict(gpu_calls)
            started=time.perf_counter()
            data=read_rect(world,'coarse',args.row,args.col,args.row+args.size,args.col+args.size)
            # Exact same normalization as terrain_server._field_block.
            field=(data[0]/(data[-1]+1e-8)).float().cpu().numpy()
            torch.cuda.synchronize()
            after=dict(gpu_calls)
            report=dict(kind='learned-coarse',seed=args.seed,profile=args.profile,row=args.row,col=args.col,
                size=args.size,capture_seconds=time.perf_counter()-started,
                gpu_calls_before=before,gpu_calls_after=after,
                gpu_calls_delta={k:after[k]-before[k] for k in before},
                inference=inference_status(world),gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,cuda=torch.version.cuda,
                generation_settings=world._terrain_generation_settings)
        finally:
            world.close()
    if not np.isfinite(field).all():
        raise ValueError('Nonfinite learned coarse crop')
    path=output/'coarse-source.npz'
    np.savez_compressed(path,sqrt_height=field)
    report['source_sha256']=sha(path)
    report['sources']={name:dict(path=str(Path(module.__file__).resolve()),sha256=sha(module.__file__))
        for name,module in sys.modules.copy().items() if name in
        ('terrain_app','terrain_inference','terrain_nn_constants','terrain_cuda_kernels',
         'terrain_coarse_graph','terrain_conditioning','terrain_window_scheduler',
         'terrain_diffusion.inference.world_pipeline')}
    save(output/'coarse-source.json',report)
    return field,report


def coordinates(tx,ty):
    offset=(np.arange(-HALO,TILE+HALO,dtype=np.float64)+.5)*STEP
    return tx*TILE*STEP+offset,ty*TILE*STEP+offset


def compare(field,source,repeats):
    if min(field.shape)<96 or not np.isfinite(field).all():
        raise ValueError('Need a finite coarse crop >=96x96 with room for four tile halos')
    physical=np.sign(field)*field**2
    margin=40
    peak=np.unravel_index(np.argmax(physical[margin:-margin,margin:-margin]),
                          physical[margin:-margin,margin:-margin].shape)
    peak=(peak[0]+margin,peak[1]+margin)
    grow,gcol=int(source['row']+peak[0]),int(source['col']+peak[1])
    tx,ty=math.floor((gcol+.5)/16)-1,math.floor((grow+.5)/16)-1
    sample,requests=sample_function(field,source['row'],source['col'])
    arrays,tiles={},[]
    timing={key:[] for key in ('bilinear_interpolation_ms','smooth_interpolation_ms','bilinear_gaussian_ms','smooth_gaussian_ms')}
    for dy in range(2):
        for dx in range(2):
            x,y=tx+dx,ty+dy
            xs,ys=coordinates(x,y)
            same_reads=[]
            for smooth,name in ((False,'bilinear'),(True,'smooth')):
                requests.clear()
                elevation=sample(None,xs,ys,'coarse',smooth_coarse=smooth)
                same_reads.append(list(requests))
                arrays[f'{name}_{dx}_{dy}_raw']=elevation
                arrays[f'{name}_{dx}_{dy}_filtered']=gaussian_filter(elevation,sigma=.65,mode='reflect').astype(np.float32)
            if same_reads[0]!=same_reads[1]:
                raise ValueError('Interpolation variants requested different source halos')
            tiles.append(dict(tx=x,ty=y,dx=dx,dy=dy,source_reads=same_reads[0]))
    # Timings include interpolation/formula plus a bounded NumPy view provider,
    # never an NN read, disk read, transfer, or image rendering.
    xs,ys=coordinates(tx+1,ty+1)
    for iteration in range(repeats):
        order=((False,'bilinear'),(True,'smooth'))
        if iteration%2:
            order=tuple(reversed(order))
        for smooth,name in order:
            started=time.perf_counter()
            elevation=sample(None,xs,ys,'coarse',smooth_coarse=smooth)
            timing[name+'_interpolation_ms'].append((time.perf_counter()-started)*1000)
            started=time.perf_counter()
            gaussian_filter(elevation,sigma=.65,mode='reflect').astype(np.float32)
            timing[name+'_gaussian_ms'].append((time.perf_counter()-started)*1000)
    seams=[]
    # Adjacent stored tiles overlap by 48 pixels. Gaussian reflect affects only
    # the outer3 pixels at sigma0.65/truncate4; exclude these in filtered checks.
    for name in ('bilinear','smooth'):
        for stage in ('raw','filtered'):
            trim=3 if stage=='filtered' else 0
            end=2*HALO-trim
            for dy in range(2):
                a,b=(arrays[f'{name}_{dx}_{dy}_{stage}'] for dx in range(2))
                error=a[:,TILE+trim:TILE+end]-b[:,trim:end]
                seams.append(dict(mode=name,stage=stage,axis='x',pair=dy,trim=trim,max_abs_m=float(np.abs(error).max())))
            for dx in range(2):
                a,b=(arrays[f'{name}_{dx}_{dy}_{stage}'] for dy in range(2))
                error=a[TILE+trim:TILE+end,:]-b[trim:end,:]
                seams.append(dict(mode=name,stage=stage,axis='y',pair=dx,trim=trim,max_abs_m=float(np.abs(error).max())))
    for name in ('bilinear','smooth'):
        for stage in ('raw','filtered'):
            rows=[np.concatenate([arrays[f'{name}_{dx}_{dy}_{stage}'][HALO:HALO+TILE,HALO:HALO+TILE]
                                   for dx in range(2)],axis=1) for dy in range(2)]
            arrays[f'{name}_{stage}_mosaic']=np.concatenate(rows,axis=0)
    for key,value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite interpolation output: {key}')
    delta=arrays['smooth_filtered_mosaic']-arrays['bilinear_filtered_mosaic']
    result=dict(peak_coarse_height_m=float(physical[peak]),peak_coarse_index=[grow,gcol],
        mosaic_origin_tile=[tx,ty],lod=LOD,resolution_m=NATIVE*STEP,source_resolution_m=NATIVE*COARSE_STRIDE,
        coarse_cell_width_display_pixels=16,tiles=tiles,same_source_halos=True,seams=seams,
        max_seam_error_m=max(x['max_abs_m'] for x in seams),
        display_change=dict(max_abs_m=float(np.abs(delta).max()),mae_m=float(np.abs(delta).mean()),
                            rmse_m=float(np.sqrt(np.mean(delta.astype(np.float64)**2)))),
        interpolation_timings={key:dict(median_ms=statistics.median(values),maximum_ms=max(values),samples_ms=values)
                               for key,values in timing.items()},
        comparison_neural_calls=0,comparison_source='sealed CPU views of one saved coarse crop',
        caveat='Deliberate preview interpolation change, not a new neural prediction or global seam proof')
    return arrays,result


def render(arrays,report,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from render_inference_comparisons import terrain_rgb,slope
    folder.mkdir(parents=True,exist_ok=True)
    before,after=(arrays[f'{name}_filtered_mosaic'] for name in ('bilinear','smooth'))
    resolution=report['resolution_m']
    norm=TwoSlopeNorm(vmin=min(-1.,float(before.min()),float(after.min())),vcenter=0,
                      vmax=max(1.,float(before.max()),float(after.max())))
    cmap=plt.get_cmap('terrain')
    images=[terrain_rgb(z,norm,cmap,resolution) for z in (before,after)]
    extent=(0,before.shape[1]*resolution/1000,before.shape[0]*resolution/1000,0)
    slopes=[slope(z,resolution) for z in (before,after)]
    max_slope=max(float(z.max()) for z in slopes)
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    fig.suptitle(f'Coarse NN réel — seed {report["source"]["seed"]} — LOD4 :480m/pixel\n'
                 f'Source coarse :7680m/cellule (16pixels) · sommet coarse {report["peak_coarse_height_m"]:.0f}m · Gaussian σ0,65 identique')
    for i,name in enumerate(('Avant : bilinéaire','Après : Hermite monotone')):
        axes[0,i].imshow(images[i],extent=extent,interpolation='nearest')
        axes[0,i].set_title(name)
        image=axes[1,i].imshow(slopes[i],extent=extent,cmap='magma',vmin=0,vmax=max_slope,interpolation='nearest')
        axes[1,i].set_title('Pente — '+name)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes[0,:2],shrink=.65,label='Altitude(m), échelle commune')
    fig.colorbar(image,ax=axes[1,:2],shrink=.65,label='Pente(°)')
    delta=after-before
    limit=max(1.,float(np.abs(delta).max()))
    image=axes[0,2].imshow(delta,extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
    axes[0,2].set_title('Variation d’affichage après−avant')
    fig.colorbar(image,ax=axes[0,2],label='m',shrink=.65)
    peak=np.unravel_index(np.argmax(before),before.shape)
    row=peak[0]
    x=np.arange(before.shape[1])*resolution/1000
    axes[1,2].plot(x,before[row],label='Avant')
    axes[1,2].plot(x,after[row],label='Après')
    axes[1,2].axvline(TILE*resolution/1000,color='black',linestyle=':',label='Jonction des tuiles')
    axes[1,2].set_title('Profil passant par le pic de la vue avant')
    axes[1,2].set_ylabel('Altitude(m)')
    axes[1,2].legend()
    for ax in axes.flat:
        ax.set_xlabel('Distance locale x(km)')
    fig.savefig(folder/'overview.png',dpi=145)
    plt.close(fig)
    y=max(0,min(peak[0]-64,before.shape[0]-128));x0=max(0,min(peak[1]-64,before.shape[1]-128))
    crop=np.s_[y:y+128,x0:x0+128]
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    fig.suptitle('Zoom128pixels près du sommet — mêmes échelles, aucun détail NN ajouté')
    for i,label in enumerate(('Avant','Après')):
        axes[0,i].imshow(images[i][crop],interpolation='nearest')
        axes[0,i].set_title(label+' — relief ombré')
        axes[1,i].imshow(slopes[i][crop],cmap='magma',vmin=0,vmax=max_slope,interpolation='nearest')
        axes[1,i].set_title(label+' — pentes')
    fig.savefig(folder/'summit-zoom.png',dpi=145)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    fig.suptitle('Raccord central de quatre tuiles — profils et dérivées locales')
    for col,axis in enumerate(('horizontal','vertical')):
        for array,label in ((before,'Avant'),(after,'Après')):
            line=array[TILE,:] if col==0 else array[:,TILE]
            axes[0,col].plot(np.arange(-32,33)*resolution/1000,line[TILE-32:TILE+33],label=label)
            derivative=np.diff(line)/resolution
            axes[1,col].plot((np.arange(-32,32)+.5)*resolution/1000,derivative[TILE-32:TILE+32],label=label)
        axes[0,col].set_title('Profil '+axis)
        axes[0,col].set_ylabel('Altitude(m)')
        axes[1,col].set_ylabel('Dérivée(m/m)')
        for ax in axes[:,col]:
            ax.axvline(0,color='black',linestyle=':')
            ax.legend()
            ax.set_xlabel('Distance à la jonction(km)')
    fig.savefig(folder/'tile-seams.png',dpi=145)
    plt.close(fig)
    text=f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Interpolation coarse LOD4</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}}img{{width:100%}}p{{max-width:1000px;line-height:1.5}}</style>
<h1>LOD4 : interpolation du même relief coarse appris</h1>
<p>Seed{report['source']['seed']}, profil {html.escape(report['source']['profile'])}. Pic coarse :{report['peak_coarse_height_m']:.1f}m.
Affichage480m/pixel ; source7680m/cellule, soit16pixels. Les deux versions lisent les mêmes halos et appliquent le même Gaussian σ0,65.
Aucun appel NN supplémentaire pour la comparaison. Le lissage ne crée pas de détail appris ; des motifs de la grille source peuvent rester.</p>
<p>Variation d’affichage max{report['display_change']['max_abs_m']:.3f}m ; RMSE{report['display_change']['rmse_m']:.3f}m.
Raccord maximal sur les recouvrements testés :{report['max_seam_error_m']:.6g}m.
Ce résultat porte sur quatre tuiles voisines de ce crop, pas tous les raccords du monde.</p>
<img src="overview.png"><img src="summit-zoom.png"><img src="tile-seams.png">
<p><a href="report.json">Provenance et métriques</a></p></html>'''
    (folder/'index.html').write_text(text,encoding='utf-8')
    save(folder/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture',action='store_true')
    mode.add_argument('--source',type=Path)
    mode.add_argument('--self-test',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--profile',default='natural')
    parser.add_argument('--row',type=int,default=0)
    parser.add_argument('--col',type=int,default=-192)
    parser.add_argument('--size',type=int,default=160)
    parser.add_argument('--min-peak-m',type=float,default=2500.)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--output',type=Path,default=RUNTIME)
    parser.add_argument('--images',type=Path,default=IMAGES)
    args=parser.parse_args()
    args.output,args.images=args.output.resolve(),args.images.resolve()
    if args.source:
        args.source=args.source.resolve()
    if not 96<=args.size<=256 or not 1<=args.repeats<=200:
        raise ValueError('Bounded size96..256 and repeats1..200 required')
    if args.self_test:
        y,x=np.mgrid[:160,:160]
        metres=4000*np.exp(-((x-83)**2+(y-79)**2)/350)+200*np.sin(x/8)*np.cos(y/7)
        field=(np.sign(metres)*np.sqrt(np.abs(metres))).astype(np.float32)
        source=dict(kind='synthetic-self-test',row=0,col=-192,seed=42,profile='synthetic')
        arrays,result=compare(field,source,3)
        if not result['same_source_halos'] or result['max_seam_error_m']>1e-4:
            raise AssertionError(result['seams'])
        json.dumps(result,allow_nan=False,default=json_scalar)
        json.dumps(dict(status='failed',value=np.int64(42)),allow_nan=False,default=json_scalar)
        print(json.dumps(dict(status='self-test-pass',max_seam_error_m=result['max_seam_error_m'],gpu=False)))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',sources={name:sha(ROOT/name) for name in
        ('validate_coarse_interpolation.py','terrain_interpolation.py','terrain_server.py','render_inference_comparisons.py')})
    save(args.output/'report.json',report)
    try:
        if args.capture:
            field,source=capture(args,args.output)
        else:
            source=json.loads(args.source.with_suffix('.json').read_text(encoding='utf-8'))
            if sha(args.source)!=source['source_sha256'] or source['kind']!='learned-coarse':
                raise ValueError('Saved learned coarse provenance mismatch')
            with np.load(args.source,allow_pickle=False) as archive:
                field=archive['sqrt_height'].copy()
        report['source']=source
        arrays,result=compare(field,source,args.repeats)
        report.update(result)
        np.savez_compressed(args.output/'comparison.npz',**arrays)
        report['arrays_sha256']=sha(args.output/'comparison.npz')
        if report['peak_coarse_height_m']<args.min_peak_m:
            raise ValueError('Selected crop has no sufficiently high mountain; inspect receipt and choose a new crop')
        if report['max_seam_error_m']>1e-4:
            raise ValueError('Local tile overlap mismatch')
        report['status']='complete'
        render(arrays,report,args.images)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save(args.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],peak_m=report['peak_coarse_height_m'],
                         gallery=str(args.images/'index.html'),comparison_neural_calls=0)))


if __name__=='__main__':
    main()
