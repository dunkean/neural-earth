"""Prepare the selected browser models without any research dataset/checkpoint.

Exports are cached outside Git and shared between the two decoder choices.
Only a completed, numerically checked bundle becomes visible to the app.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from backend.terrain_paths import REPO_ROOT, RUNTIME_ROOT, HF_HOME, HF_HUB_CACHE, model_snapshot


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def bundle_identity():
    files=['distill/weights/manifest.json','distill/tile40_prepare.py','distill/student.py',
        'distill/features.py','distill/tile40_flow_network.py','distill/tile40_flow_runtime.py',
        'distill/tile40_runtime.py','distill/tile40_height.py','distill/tile40_export_utils.py',
        'webgpu/tile40_runtime.mjs','webgpu/noise.mjs']
    return hashlib.sha256(json.dumps({name:sha(REPO_ROOT/name) for name in files},sort_keys=True).encode()).hexdigest()[:20]


def bundle_root():
    return RUNTIME_ROOT/'browser-models'/('tile40-'+bundle_identity())


def prepare(output=None):
    os.environ.setdefault('HF_HOME',str(HF_HOME))
    os.environ.setdefault('HF_HUB_CACHE',str(HF_HUB_CACHE))
    os.environ.setdefault('TERRAIN_RUNTIME_ROOT',str(RUNTIME_ROOT))
    from distill.common import bootstrap,atomic_json,atomic_write
    bootstrap()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from distill.features import COND_MEANS,COND_STDS,base_features,decoder_features
    from distill.student import load_student
    from distill.tile40_conditioning import CLIMATES,sketch
    from distill.tile40_export_utils import compact_decoder_input,EncoderRegion,ExportCoarse
    from distill.tile40_flow_runtime import FiniteFlowEncoder
    from distill.tile40_runtime import Tile40,coarse_input
    from distill.tile40_conditioning import SketchConditioning
    output=Path(output or bundle_root()).resolve()
    if output==REPO_ROOT or REPO_ROOT in output.parents:
        raise ValueError('Browser exports belong outside Git.')
    if (output/'catalog.json').is_file():
        return output
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<2*2**30:
        raise ValueError('Keep at least2GiB free before preparing browser models.')
    torch.set_num_threads(8)
    torch.manual_seed(20261020)
    source=REPO_ROOT/'distill/weights'
    distribution=json.loads((source/'manifest.json').read_text())
    for item in distribution['exports'].values():
        if sha(source/item['file'])!=item['sha256']:
            raise ValueError('Selected weights differ from the distribution manifest.')
    revision=distribution['teacher_revision']
    config_path=model_snapshot(revision)/'config.json'
    if not config_path.is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=distribution['teacher_model'],revision=revision,
            cache_dir=str(HF_HUB_CACHE),allow_patterns=['config.json','coarse_model/*'])
    config=json.loads(config_path.read_text())
    keys=('cond_snr','frequency_mult','coarse_means','coarse_stds','histogram_raw',
        'residual_mean','residual_std','coarse_pooling','drop_water_pct','elev_coarse_pool_mode','p5_coarse_pool_mode')
    parameters={key:config[key] for key in keys}
    parameters['histogram_raw']=parameters['histogram_raw'] or [0.]*5
    coarse,_=load_student(source/'coarse2.pt')
    coarse=ExportCoarse(coarse.solver).eval().requires_grad_(False)
    encoder,saved=load_student(source/'base-residual8k.pt')
    if saved['contract']['coarse_sha256']!=distribution['exports']['coarse2']['sha256']:
        raise ValueError('Residual8k requires its pinned coarse2 weights.')
    encoder=FiniteFlowEncoder(encoder,saved['contract'],steps=4).eval().requires_grad_(False)
    options=ort.SessionOptions();options.intra_op_num_threads=8

    def export(model,inputs,path,output_name,tolerance):
        with torch.no_grad():expected=model(inputs).numpy()
        torch.onnx.export(model,inputs,str(path),input_names=['inputs'],output_names=[output_name],
            opset_version=17,dynamo=False,do_constant_folding=True,external_data=False)
        graph=onnx.load(str(path))
        for node in graph.graph.node:
            attrs={a.name:a for a in node.attribute}
            if node.op_type=='Conv' and len(attrs['kernel_shape'].ints)==2 and len(attrs['pads'].ints)==2:
                pads=list(attrs['pads'].ints);attrs['pads'].ints[:]=pads+pads
        onnx.checker.check_model(graph);onnx.save(graph,str(path))
        if path.stat().st_size>=100000000:raise ValueError('A browser model exceeds100MB.')
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        actual=session.run(None,{'inputs':inputs.numpy()})[0]
        delta=float(np.abs(actual-expected).max())
        if not np.isfinite(actual).all() or delta>tolerance:
            raise ValueError(f'Numerical export disagreement: {path.name}: {delta}')
        del session,expected,actual,graph
        gc.collect()
        return dict(file=path.name,output_name=output_name,sha256=sha(path),bytes=path.stat().st_size,
                    input_shape=list(inputs.shape),export_max_abs=delta)

    with tempfile.TemporaryDirectory(prefix='.tile40-',dir=output.parent) as temporary:
        work=Path(temporary)
        factory=SketchConditioning(sketch(9100000,'mountains'),'arid')
        coarse_inputs=coarse_input(factory,parameters,9100000,-48,-48)
        coarse_info=export(coarse,coarse_inputs,work/'coarse-f32.onnx','coarse',.001)
        coarse_info.update(checkpoint_sha256=distribution['exports']['coarse2']['sha256'],step=200000)
        prepared_parameters=dict(parameters)
        theta=torch.atan(torch.tensor(parameters['cond_snr'])).bfloat16()
        prepared_parameters.update(cos_theta=torch.cos(theta).float().tolist(),sin_theta=torch.sin(theta).float().tolist(),
            labels=torch.log(torch.tan(theta)/8).float().tolist())
        means=np.array(parameters['coarse_means'],np.float32);stds=np.array(parameters['coarse_stds'],np.float32)
        means[1],stds[1]=means[0]-means[1],stds[0]
        prepared_parameters.update(physical_means=means.tolist(),physical_stds=stds.tolist())
        axis=torch.arange(-5,6,dtype=torch.float32);laplacian=torch.exp(-.5*(axis/5.).square());laplacian/=laplacian.sum()
        guidance=np.exp(-.5*(np.arange(-32,33,dtype=np.float64)/8.)**2);guidance/=guidance.sum()
        encoder_info=None
        for key,whole in [('decoder16',True),('decoder64',False)]:
            directory=work/key;directory.mkdir()
            decoder_path=source/distribution['exports'][key]['file']
            decoder,decoder_saved=load_student(decoder_path)
            generator=Tile40(coarse,encoder,decoder,parameters,whole_decoder=whole)
            actual=generator.generate(factory.elevation,'arid',9100000,trace=True)
            if encoder_info is None:
                inputs=base_features(actual['coarse'],-8,-8,9100000,-64,-64,320,parameters['histogram_raw'])[None]
                encoder_info=export(EncoderRegion(encoder),inputs,work/'encoder-f32.onnx','latents',.0005)
                encoder_info.update(checkpoint_sha256=distribution['exports']['base_residual8k']['sha256'],step=8000,selected_weights='raw')
            if whole:
                fields=torch.from_numpy(actual['latents'][:4,32:264,32:264])
                fixture=decoder_features(fields,9100000,-256,-256,1856,noise_tile=512)[None,:5]
            else:
                fields=torch.from_numpy(actual['latents'][:4,64:128,64:128])
                fixture=decoder_features(fields,9100000,0,0)[None,:5]
            decoder_info=export(compact_decoder_input(decoder),fixture,directory/'decoder-f32.onnx','residual',.0001)
            decoder_info.update(input_channels=5,layout='finite-root' if whole else 'overlap512',
                checkpoint_sha256=distribution['exports'][key]['sha256'],step=decoder_saved['step'],selected_weights='ema')
            for item in [coarse_info,encoder_info]:os.link(work/item['file'],directory/item['file'])
            models=dict(coarse=coarse_info,encoder=encoder_info,decoder=decoder_info)
            atomic_json(directory/'input.json',dict(sketch_m=factory.elevation.tolist(),climate='arid',seed=9100000))
            atomic_write(directory/'expected-height.bin',lambda f:f.write(actual['height_m'].astype('<f4').tobytes()))
            manifest=dict(schema=1,models=models,total_model_bytes=sum(v['bytes'] for v in models.values()),
                encoder_halo=0,coarse_origin=-8,coarse_size=24,parameters=prepared_parameters,
                condition_means=COND_MEANS.tolist(),condition_stds=COND_STDS.tolist(),climates=CLIMATES,
                kernels=dict(laplacian=laplacian.tolist(),guidance=guidance.tolist()),
                side_m=40000.,pixels=1334,spacing_m=40000./1334,
                input_file='input.json',input_sha256=sha(directory/'input.json'),
                expected_file='expected-height.bin',expected_sha256=sha(directory/'expected-height.bin'),
                sampling=dict(kind='conditional-flow',evaluations=8,intervals=4,finite_tile=True),
                selected_models=dict(coarse='coarse2',base='base_residual8k',decoder=key),
                scope='Finite40km tile from sketch/climate/seed;experimental quality.',
                provenance=dict(runtime_sha256=sha(REPO_ROOT/'webgpu/tile40_runtime.mjs'),noise_sha256=sha(REPO_ROOT/'webgpu/noise.mjs')))
            atomic_json(directory/'manifest.json',manifest)
            snapshot=directory/'runtime-snapshot';snapshot.mkdir()
            for name in ['tile40_runtime.mjs','noise.mjs']:shutil.copy2(REPO_ROOT/'webgpu'/name,snapshot/name)
            del generator,decoder,actual,fixture
            gc.collect()
        atomic_json(work/'catalog.json',dict(status='ready',default_decoder='decoder16',base='Résiduel48 · 8k',
            coarse='Coarse2 · 200k',decoders=[dict(id='decoder16',label='Décodeur16 · 10k · compact'),
                                         dict(id='decoder64',label='Décodeur64 · 200k · qualité')]))
        for name in ['coarse-f32.onnx','encoder-f32.onnx']:(work/name).unlink()
        os.replace(work,output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    from filelock import FileLock,Timeout
    from distill.common import atomic_json
    output=Path(args.output or bundle_root())
    output.parent.mkdir(parents=True,exist_ok=True)
    status=output.with_suffix('.status.json')
    try:
        with FileLock(str(output)+'.lock',timeout=0):
            atomic_json(status,dict(status='preparing',pid=os.getpid()))
            try:
                print(prepare(output),flush=True)
            except Exception as exc:
                atomic_json(status,dict(status='error',error=str(exc)))
                raise
            atomic_json(status,dict(status='ready'))
    except Timeout:
        print('Browser models are already being prepared.',flush=True)


if __name__=='__main__':main()
