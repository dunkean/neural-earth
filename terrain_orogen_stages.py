"""Independent, immutable relief -> erosion -> climate generation artifacts.

Stage commands restore the original graph, rather than generating it again.
Composition retains untouched raster fields, even when their inputs are stale.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
from terrain_generation_session import check_generation, cancellable_process, run_process
import tempfile
import time

import numpy as np
import torch

from terrain_generation import (OROGEN_PARAMETERS, OROGEN_CLIMATE_PARAMETERS,
    OROGEN_GPU_PARAMETERS, OROGEN_STAGE_PARAMETERS, CITY_EROSION_PARAMETERS)

VERSION = 'orogen-independent-stages-v1'
EROSION_KEYS = {'relief_pipeline', 'orogen_warp', 'orogen_smoothing',
    'orogen_hydraulic', 'orogen_thermal', 'orogen_glacial', 'orogen_sharpening',
    'orogen_gpu_post', 'orogen_gpu_erosion'} | set(CITY_EROSION_PARAMETERS)
CLIMATE_KEYS = set(OROGEN_CLIMATE_PARAMETERS) | {'climate_source',
    'orogen_temperature_offset', 'orogen_precipitation_offset', 'orogen_gpu_climate'}
RELIEF_KEYS = (set(OROGEN_PARAMETERS)-EROSION_KEYS-CLIMATE_KEYS) | {
    'world_diameter_km', 'world_topology',
    'height_source', 'continental_style', 'continental_strength', 'macro_scale_km',
    'frequency_mult', 'octaves', 'drop_water_pct', 'orogen_gpu_relief',
    'orogen_gpu_propagation', 'orogen_gpu_raster'}
STAGE_KEYS = dict(relief=RELIEF_KEYS, erosion=EROSION_KEYS, climate=CLIMATE_KEYS)
EROSION_FIELDS = {'erosionDelta', 'city_erosion_delta_m'}
CLIMATE_FIELDS = {'koppen', 'pressureSummer', 'pressureWinter', 'continentality',
                  'rainShadowSummer', 'rainShadowWinter'}


def is_climate_field(name):
    return name in CLIMATE_FIELDS or name.startswith(('temperature_', 'precip_',
        'wind_', 'ocean_', 'biome_', 'koppen_color_'))


def _settings(config, style):
    from terrain_generation import _defaults
    source=config['initial_source']
    base='orogen' if source=='orogen' else 'natural' if source=='natural' else 'terrestrial-'+style
    settings=_defaults(base)
    settings.update(config['initial_settings'])
    settings.update({k:config[k.removeprefix('orogen_')] for k in OROGEN_PARAMETERS})
    settings.update({k:config[k] for k in (*CITY_EROSION_PARAMETERS,*OROGEN_GPU_PARAMETERS)})
    settings.update(height_source=source,relief_pipeline=config['relief_pipeline'],
                    continental_style=style)
    return settings


def _path(key):
    import terrain_orogen as core
    if not isinstance(key,str) or re.fullmatch('[0-9a-f]{64}',key) is None:
        raise ValueError('Invalid generation stage identity')
    return core.CACHE_ROOT/core.VERSION/'stages'/(key+'.npz')


def load_stage(key, stage=None, *, seed=None, width=None, height=None):
    try:
        with np.load(_path(key),allow_pickle=False) as saved:
            meta=json.loads(str(saved['metadata']))
            arrays={name:saved[name] for name in meta['sha256']}
        if hashlib.sha256(_json(meta['namespace'])).hexdigest()!=key:
            raise ValueError('Stage identity mismatch')
        for name,value in arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest()!=meta['sha256'][name]:
                raise ValueError('Stage data checksum mismatch')
            value.flags.writeable=False
        ns=meta['namespace']
        for name,wanted in [('stage',stage),('seed',seed),('width',width),('height',height)]:
            if wanted is not None and ns[name]!=wanted:
                raise ValueError('Stage '+name+' mismatch')
        shape=(ns['height'],ns['width'])
        for name,value in arrays.items():
            if name=='height_m' or name.startswith('layer_'):
                if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
                    raise ValueError('Invalid stage raster')
        return dict(id=key,metadata=meta,arrays=arrays)
    except (OSError,ValueError,KeyError) as error:
        raise ValueError('Cannot load '+str(stage or '')+' generation stage: '+str(error)) from error


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _save(key,namespace,arrays,details):
    import terrain_orogen as core
    metadata=dict(namespace=namespace,sha256={k:hashlib.sha256(v.tobytes()).hexdigest()
                  for k,v in arrays.items()},**details)
    stream=io.BytesIO()
    np.savez(stream,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),**arrays)
    core._atomic(_path(key),stream.getvalue())
    return load_stage(key,namespace['stage'])


def _request(seed,config):
    digest=hashlib.sha256(f'orogen-tectonic-cuda-v1:{seed}'.encode()).digest()
    sparse=seed if seed<=0x7fffffff else int.from_bytes(digest[:4],'little')&0x7fffffff
    return dict(seed=sparse,N=config['detail'],P=config['plate_count'],jitter=.75,
        nMag=config['roughness'],numContinents=config['continent_count'],
        continentSizeVariety=config['continent_variety'],landCoverage=config['land_coverage'],
        smoothing=config['smoothing'],terrainWarp=config['warp'],hydraulicErosion=config['hydraulic'],
        thermalErosion=config['thermal'],glacialErosion=config['glacial'],ridgeSharpening=config['sharpening'],
        temperatureOffset=config['temperature_offset'],precipitationOffset=config['precipitation_offset'],
        threshold=config['convergence_threshold'],motion=config['motion_strength'],spread=config['spread'],
        climateSettings={k.removeprefix('orogen_'):config[k.removeprefix('orogen_')]
                         for k in OROGEN_CLIMATE_PARAMETERS})


def _execute(stage, seed, style, config, width, height, parent=None, relief=None):
    import terrain_orogen as core
    from terrain_orogen_cuda import rasterize
    begun=time.perf_counter()
    identity=core.implementation_identity()
    request=_request(seed,config)
    flags={key:False for key in ('relief','propagation','post','erosion','climate')}
    for key in {'relief':('relief','propagation'),'erosion':('post','erosion'),'climate':('climate',)}[stage]:
        flags[key]=config['orogen_gpu_'+key]
    gpu=None;erosion=None
    device=torch.device('cuda',identity['cuda_device']['index'])
    with tempfile.TemporaryDirectory(prefix='orogen-'+stage+'-') as tmp:
        directory=Path(tmp)
        snapshot=directory/'snapshot.bin'
        if stage!='climate':request['snapshotOutput']=str(snapshot)
        base=None
        if stage=='relief':
            request.update(skipPost=True,skipClimate=True)
            base=core.initial_raster(seed,style,config,width,height)
            if base is not None:
                np.asarray(base,dtype='<f4').tofile(directory/'initial.f32')
                request.update(inputFile=str(directory/'initial.f32'),imageWidth=width,imageHeight=height)
        else:
            retained=directory/'retained.bin'
            retained.write_bytes(parent['arrays']['snapshot'].tobytes())
            request.update(resumeFile=str(retained),resumeCommand=stage,
                           legacyHeightConvention=parent['metadata'].get('height_convention')!=core.HYPSOMETRY_RULE,
                           imported=relief['metadata']['namespace']['settings']['height_source']!='orogen',
                           skipPost=config['relief_pipeline']!='orogen',skipClimate=stage=='erosion')
        geometry=relief['arrays'] if relief is not None else None
        city=stage=='erosion' and config['relief_pipeline']=='city-gpu' and config['city_erosion_strength']>0
        city_result={}
        def city_callback():
            from terrain_city_erosion import erode
            from terrain_geometry import world_bounds
            extent=world_bounds(_settings(config,style))
            eroded,meta=erode(parent['arrays']['height_m'],world_height_m=extent[3]-extent[1],strength=config['city_erosion_strength'],
                iterations=config['city_erosion_iterations'],talus=config['city_erosion_talus'],
                motif_km=config['city_erosion_motif_km'])
            _,p,_,f=core._graph_fields(directory)
            core._atlas_to_elevation(eroded,p,f['elevation']).tofile(directory/'external.f32')
            city_result.update(height=eroded,metadata=meta)
            return dict(elevationFile=str(directory/'external.f32')),meta
        if city:request['externalErosion']=True
        if config.get('orogen_gpu_identity',{}).get('available') and any(flags.values()):
            from terrain_orogen_gpu import run
            gpu,_=run(shutil.which('node'),directory,request,flags,city_callback if city else None)
        elif city:
            # Reuse the existing framed bridge even with no adapted CUDA loops.
            # It is only orchestration here; City remains the WGSL engine.
            if config.get('orogen_gpu_identity',{}).get('available'):
                from terrain_orogen_gpu import run
                gpu,_=run(shutil.which('node'),directory,request,flags,city_callback)
            else:
                request_path=directory/'request.json';request_path.write_bytes(_json(request))
                with (directory/'node.log').open('w+b') as log:
                    with cancellable_process([shutil.which('node'),str(core.NATIVE/'pipeline.mjs'),str(request_path),str(directory)],
                         stdin=subprocess.PIPE,stdout=log,stderr=log) as process:
                        while not (directory/'relief.ready').exists():
                            if process.poll() is not None or time.perf_counter()-begun>600:
                                process.kill();log.seek(0)
                                raise RuntimeError('Retained City preparation failed: '+log.read().decode(errors='replace')[-4000:])
                            time.sleep(.02)
                        reply,_=city_callback()
                        process.communicate(_json(reply),timeout=600)
                        if process.returncode:
                            log.seek(0);raise RuntimeError(log.read().decode(errors='replace')[-4000:])
        else:
            request_path=directory/'request.json';request_path.write_bytes(_json(request))
            result=run_process([shutil.which('node'),str(core.NATIVE/'pipeline.mjs'),str(request_path),str(directory)],
                                  capture_output=True,text=True,timeout=600)
            if result.returncode:raise RuntimeError('Orogen '+stage+' failed: '+result.stderr[-6000:])
        graph,points,triangles,fields=core._graph_fields(directory)
        if stage=='relief':
            xyz=core.sphere_raster(width,height)
            nearest=core._nearest_regions(points,xyz,config)
            geometry=dict(points=points,triangles=triangles.astype(np.int32),nearest=nearest.astype(np.int32))
        else:xyz=core.sphere_raster(width,height);nearest=geometry['nearest'].copy()
        selected={k:v for k,v in fields.items() if
            (is_climate_field(k) if stage=='climate' else
             k in EROSION_FIELDS or k=='elevation' if stage=='erosion' else
             not is_climate_field(k) and k not in EROSION_FIELDS)}
        projected,fallback=rasterize(points,triangles,nearest,xyz,selected,device)
        layers={k:v.reshape(height,width) for k,v in projected.items()}
        if stage=='relief':
            raw=core._metres(layers.pop('elevation')) if base is None else np.asarray(base,np.float32).copy()
            layers['uplift']=np.maximum(layers.get('tectonic',np.zeros_like(raw)),0).astype(np.float32)
        elif stage=='erosion':
            raw=core._metres(layers.pop('elevation')) if config['relief_pipeline']=='orogen' else parent['arrays']['height_m'].copy()
            if city:
                raw=city_result['height'];erosion=city_result['metadata']
                layers['city_erosion_delta_m']=(raw-parent['arrays']['height_m']).astype(np.float32)
        else:raw=parent['arrays']['height_m']
        arrays={**{'layer_'+k:v for k,v in layers.items()}}
        if stage!='climate':
            arrays.update(height_m=raw,snapshot=np.frombuffer(snapshot.read_bytes(),np.uint8).copy())
        else:
            arrays['layer_climate_height_m']=np.asarray(raw,np.float32).copy()
        if stage=='relief':arrays.update(geometry)
    return arrays,dict(height_convention=core.HYPSOMETRY_RULE,seconds=time.perf_counter()-begun,original_pipeline=graph['timing'],
        original_elevation_timing=graph.get('elevationTiming',[]),original_post_timing=graph.get('postTiming',[]),
        gpu_pipeline=gpu,erosion=erosion,projection_fallback_pixels=fallback)


def build_stage(stage,seed,style,config,width,height,parent=None,relief=None):
    import terrain_orogen as core
    check_generation()
    settings=_settings(config,style)
    runtime={}
    gpu_keys={'relief':('relief','propagation','raster'),'erosion':('post','erosion'),'climate':('climate',)}[stage]
    if any(config['orogen_gpu_'+k] for k in gpu_keys):runtime['gpu']=config.get('orogen_gpu_identity')
    if stage=='erosion' and config['relief_pipeline']=='city-gpu':runtime['city']=config.get('city_erosion_identity')
    namespace=dict(version=VERSION,implementation=core.implementation_identity(),stage=stage,
        seed=seed,style=style if stage=='relief' else None,width=width,height=height,
        parent=parent['id'] if parent else None,
        settings={k:settings[k] for k in sorted(STAGE_KEYS[stage])},runtime=runtime)
    key=hashlib.sha256(_json(namespace)).hexdigest()
    try:return load_stage(key,stage,seed=seed,width=width,height=height),False
    except ValueError:pass
    arrays,details=_execute(stage,seed,style,config,width,height,parent,relief)
    check_generation()
    return _save(key,namespace,arrays,details),True


def compose(seed,style,config,width,height,stages,height_stage):
    import terrain_orogen as core
    relief,erosion,climate=(stages.get(s) for s in ('relief','erosion','climate'))
    active=erosion if height_stage=='erosion' and erosion else relief
    raw=active['arrays']['height_m']
    layers={k[6:]:v for k,v in relief['arrays'].items() if k.startswith('layer_')}
    if erosion:
        layers.update({k[6:]:v for k,v in erosion['arrays'].items() if k.startswith('layer_')})
    if climate:
        layers.update({k[6:]:v for k,v in climate['arrays'].items() if k.startswith('layer_')})
        if 'climate_height_m' not in layers:
            # Older retained climates need their own generating DEM, even if
            # relief/erosion have since changed independently.
            parent_id=climate['metadata']['namespace']['parent']
            parent=next((s for s in stages.values() if s['id']==parent_id),None)
            if parent is None:
                parent=load_stage(parent_id,seed=seed,width=width,height=height)
            layers['climate_height_m']=parent['arrays']['height_m']
    else:
        for name in ('temperature_summer','temperature_winter','precip_summer','precip_winter',
                     'koppen',*[prefix+'_'+axis+'_'+season for prefix in ('wind','ocean_current')
                     for axis in ('east','north') for season in ('summer','winter')],
                     *['biome_'+str(i) for i in range(3)],*['koppen_color_'+str(i) for i in range(3)]):
            layers[name]=np.zeros_like(raw)
    for name in ('boundaries','convergence','uplift'):layers.setdefault(name,np.zeros_like(raw))
    settings=_settings(config,style)
    from terrain_soil import generate_soil, VERSION as SOIL_VERSION
    from terrain_geometry import world_bounds
    layers.update(generate_soil(raw,layers,seed,world_bounds(settings),
                                settings.get('world_topology','sphere')!='plane'))
    state=dict(height_stage=height_stage,active_height=active['id'])
    for name,artifact in stages.items():
        ns=artifact['metadata']['namespace'];wanted={k:settings[k] for k in sorted(STAGE_KEYS[name])}
        dependency=relief['id'] if name=='erosion' else active['id'] if name=='climate' else None
        state[name]=dict(id=artifact['id'],parent=ns['parent'],
                         stale=ns['parent']!=dependency,settings_pending=ns['settings']!=wanted)
    metadata=dict(requested_seed_u64=str(seed),selected_seed_u64=str(seed),selected_attempt=0,
        style=style,generator_version=core.VERSION,native_config=dict(config,backend='original-Orogen; independent cached stages'),
        attempts=[],raster=dict(width=width,height=height,dtype='<f4',projection='equirectangular',samples='pixel-centres',
        y_axis='south',sea_level_m=0,longitude_axis='atan2(z,x)',reconstruction='spherical-bilinear-at-read'),
        hypsometry=dict(rule=active['metadata'].get('height_convention',
            'original Orogen: ocean 10000*e; land 6000*t^4*(5-4*t)')),sign_preserved=True,
        substrate=dict(version=SOIL_VERSION,composition=['sand','clay','humus'],
                       interpretation='procedural appearance; not surveyed geology'),
        stage_state=state,generation_seconds=0.,timings={},projection_fallback_pixels=sum(
            a['metadata']['projection_fallback_pixels'] for a in stages.values()),
        original_pipeline=[item for a in stages.values() for item in a['metadata']['original_pipeline']],
        original_elevation_timing=relief['metadata']['original_elevation_timing'],
        original_post_timing=erosion['metadata']['original_post_timing'] if erosion else [],
        diagnostics=dict(land_fraction=float(np.mean(raw>0)),highland_land_fraction=float(np.sum(raw>2000)/max(1,np.sum(raw>0)))))
    for artifact in reversed(list(stages.values())):
        for name in ('gpu_pipeline','erosion'):
            if artifact['metadata'].get(name):metadata.setdefault(name,artifact['metadata'][name])
    return raw,raw<0,metadata,layers['uplift'],layers


def generate_staged_atlas(seed,style='earthlike',*,width,height,include_layers=False,options=None):
    import terrain_orogen as core
    config=core.style_config(style,options);begun=time.perf_counter()
    stages={};execution={}
    if any(config['orogen_'+s+'_stage'] for s in STAGE_KEYS):
        for stage in STAGE_KEYS:
            key=config['orogen_'+stage+'_stage']
            if key:stages[stage]=load_stage(key,stage,seed=seed,width=width,height=height)
        if 'relief' not in stages:raise ValueError('A relief stage is required')
        height_stage=config['orogen_height_stage']
    else:
        for stage in STAGE_KEYS:
            parent=stages.get('relief' if stage=='erosion' else 'erosion') if stage!='relief' else None
            stages[stage],execution[stage]=build_stage(stage,seed,style,config,width,height,parent,stages.get('relief'))
        height_stage='erosion'
    result=compose(seed,style,config,width,height,stages,height_stage)
    result[2].update(generation_seconds=time.perf_counter()-begun,stage_execution=execution)
    return result if include_layers else result[:4]


def run_generation(seed,base,settings,stage,source_profile=None):
    """Prepare stage artifacts; register the resulting composition separately."""
    import terrain_orogen as core
    from terrain_generation import _normalize,_descriptor,resolve_generation
    if stage not in (*STAGE_KEYS,'all','settings'):raise ValueError('Unknown generation stage')
    requested=_normalize(base,settings)
    source=None
    if source_profile and stage!='all':
        previous=resolve_generation(source_profile)
        if previous.bootstrap_generator=='orogen':
            source=core.get_heightmap(seed,previous.bootstrap_style,options=previous.bootstrap_options)
    if stage not in ('all','settings') and source is None:
        raise ValueError('Generate the complete world before generating an individual stage')
    if stage=='all':
        applied=requested;stages={};height_stage='erosion'
    else:
        applied=previous.settings if source is not None else requested
        keys=STAGE_KEYS[stage] if stage in STAGE_KEYS else set(requested)-set().union(*STAGE_KEYS.values())-set(OROGEN_STAGE_PARAMETERS)
        applied.update({key:requested[key] for key in keys})
        state=source.metadata['stage_state'] if source is not None else {}
        stages={name:load_stage(state[name]['id'],name,seed=seed,width=core.WIDTH,height=core.HEIGHT)
                for name in STAGE_KEYS if name in state}
        height_stage=state.get('height_stage','erosion')
    for key,spec in OROGEN_STAGE_PARAMETERS.items():applied[key]=spec['default']
    descriptor=_descriptor('stage-build',base,applied)
    config=core.style_config(descriptor.bootstrap_style,descriptor.bootstrap_options)
    core.verify_implementation_identity()
    execution={}
    with core._LOCK:
        for name in STAGE_KEYS if stage=='all' else (stage,) if stage in STAGE_KEYS else ():
            parent=stages.get('relief' if name=='erosion' else height_stage) if name!='relief' else None
            check_generation()
            stages[name],execution[name]=build_stage(name,seed,descriptor.bootstrap_style,config,
                core.WIDTH,core.HEIGHT,parent,stages.get('relief'))
            if name in ('relief','erosion'):height_stage=name
        if stage=='all':height_stage='erosion'
    for name,artifact in stages.items():applied['orogen_'+name+'_stage']=artifact['id']
    applied['orogen_height_stage']=height_stage
    return applied,execution
