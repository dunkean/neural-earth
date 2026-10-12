"""Preserve teacher/QAT500 weight-only starts and verify their packed forwards."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory, WindowBank
from distill.native_weight_quant import (NativeWeightQuantBase, export_packed, load_packed,
                                        load_weight_quant, file_sha)
from terrain_diffusion.models.edm_unet import EDMUnet2D


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=('teacher','qat500'),default='teacher')
    parser.add_argument('--scheme',choices=('nf4','int8'),default='nf4')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-gpu',action='store_true')
    args=parser.parse_args();output=external_path(args.output);torch.set_num_threads(8)
    if (output/'status.json').exists():raise ValueError('Preserve the existing initialization.')
    teacher_path=teacher_source()/'diffusion_pytorch_model.safetensors'
    if args.source=='teacher':
        teacher=EDMUnet2D.from_pretrained(str(teacher_source())).eval()
        config=dict(teacher.config);state={'net.'+n:v.bfloat16() for n,v in teacher.state_dict().items()}
        source_hash=file_sha(teacher_path);del teacher
    else:
        source_path=DATA/'ckpt/native-fp8-pilot/step000500.pt'
        source=torch.load(source_path,map_location='cpu',weights_only=True,mmap=True)
        if source['step']!=500 or source['architecture']!='native-fp8':raise ValueError('Immutable QAT500 required.')
        config=source['config'];state={n:v.bfloat16() for n,v in source['ema'].items() if not n.endswith('._fp8_scale')}
        source_hash=file_sha(source_path);del source
    model=NativeWeightQuantBase(config,args.scheme,512).eval();model.load_state_dict(state)
    contract=dict(source=args.source,source_sha256=source_hash,scheme=args.scheme,block=512,
        source_sha256_code={n:file_sha(REPO/'distill'/n) for n in ('native_weight_quant.py','native_weight_quant_prepare.py')},
        stored_sensitive_dtype='bfloat16',scale_dtype='float32',activations='native BF16 without activation quantization',
        full_native_geometry='192x3,64/32,T2,scalar58',accepted=False,accelerated_arithmetic=False)
    saved=dict(config=config,ema=state,architecture='native-weight-quant',step=0,scheme=args.scheme,block=512,
        quantized_layers=model.quantized_layers,contract=contract,teacher_revision=REVISION,
        teacher_weights_sha256=file_sha(teacher_path),evaluation_only=True,accepted=False)
    atomic_write(output/'initial.pt',lambda f:torch.save(saved,f));atomic_json(output/'contract.json',contract)
    del model;gc.collect()
    manifest=export_packed(saved,output/'packed',device='cuda' if args.verify_gpu else 'cpu')
    print(json.dumps(dict(packed_Mo=manifest['bytes']/1e6)),flush=True)
    if args.scheme=='nf4' and manifest['bytes']>150_000_000:
        raise ValueError('Actual NF4 packed tensor file exceeds the150Mo fallback budget.')
    proof=[]
    if args.verify_gpu:
        torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        records,_,_=inventory();bank=WindowBank(records)
        simulation=load_weight_quant(saved,'cuda');packed=load_packed(output/'packed','cuda')
        with torch.no_grad(),simulation.cached_weights():
            for step in range(3):
                batch,selected=bank.batch(records,np.random.default_rng(240210+step),step)
                expected=simulation(batch['inputs'],batch['labels'],[batch['condition']])
                actual=packed(batch['inputs'],batch['labels'],[batch['condition']])
                if not torch.equal(actual,expected):
                    raise ValueError('Packed native forward differs from the QAT computation: '+str(float((actual-expected).abs().max())))
                if not actual.isfinite().all():raise FloatingPointError('Nonfinite packed output.')
                proof.append(dict(max_abs=0.,finite=True,worlds=[r['metadata']['seed'] for _,r in selected],
                                  coarses=[r['coarse_variant'] for _,r in selected]))
    atomic_json(output/'status.json',dict(status='complete',source=args.source,scheme=args.scheme,
        packed_bytes=manifest['bytes'],packed_forward_checks=proof,
        gpu=torch.cuda.get_device_name() if args.verify_gpu else None,accepted=False))
    print(json.dumps(dict(status='complete',source=args.source,scheme=args.scheme,checks=len(proof))),flush=True)


if __name__=='__main__':main()
