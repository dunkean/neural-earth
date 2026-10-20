"""Reallocate spare storage to the native base's final spatial convolutions.

No new optimizer updates. Recover dense trained masters,prioritize kernels
nearest the output at the highest spatial resolution,and measure actual bytes.
"""
import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from distill.common import atomic_json,external_path
from distill.native_base import teacher_source
from distill.native_weight_quant import file_sha
from distill.tile40_native3 import Native3Base,initial_scale,export_packed,load_packed
from distill.tile40_native3_bank import ClimateBank


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=external_path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    source=args.source_root/'latest.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    if saved['step']!=6000:raise ValueError('Completed6000step dense masters required.')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    config=json.loads((teacher_source()/'config.json').read_text())
    bank=ClimateBank(args.dataset)
    batch=bank.batch(np.random.default_rng(99155),'cuda',split='val')
    source_sha=file_sha(source)
    records=[]
    for selection in ('raw','ema'):
        net=EDMUnet2D.from_config(config).cuda().eval()
        model=Native3Base(net).eval().requires_grad_(False)
        model.load_state_dict(saved['model' if selection=='raw' else 'ema'])
        baseline=args.source_root/'packed'/f'step-006000-{selection}'/'manifest.json'
        previous=json.loads(baseline.read_text())
        candidates=[name for name,spec in model.quant_specs.items()
            if name.startswith('dec.512x512') and spec['bits']==3]
        extra=0;promoted=[]
        for name in reversed(candidates):
            weight=model.net.get_parameter(name)
            increase=math.ceil(weight.numel()/model.block)*model.block//8
            if previous['bytes']+extra+increase+65536>=100000000:continue
            spec=model.quant_specs[name]
            spec['bits']=4
            getattr(model,spec['scale_name']).copy_(initial_scale(weight,4,model.block).float())
            extra+=increase;promoted.append(name)
        if not promoted:raise ValueError('No final-resolution kernel fits the byte budget.')
        output=root/'packed'/f'step-006000-{selection}'
        packed=export_packed(model,output,step=6000,selection=selection,header_reserve=65536,
            contract=dict(source_full_checkpoint_sha256=source_sha,source_baseline_sha256=previous['sha256'],
                precision_variant='Prioritize final spatial convolutions,nearest output first,no optimizer updates.',
                promoted_to4bit=promoted,source_training_contract_sha256=file_sha(args.source_root/'contract.json')))
        loaded,_=load_packed(output,'cuda')
        with model.cached_weights():
            expected=model(batch['inputs'],batch['labels'],[batch['condition']])
        actual=loaded(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
            conditional_inputs=[batch['condition'].bfloat16()]).float()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        records.append(dict(selection=selection,artifact_bytes=packed['total_artifact_bytes'],extra_code_bytes=extra,
            promoted_kernels=promoted,packed_forward_exact=True,packed=str(output)))
        print(json.dumps(records[-1]),flush=True)
        del model,net,loaded;gc.collect();torch.cuda.empty_cache()
    atomic_json(root/'export-proof.json',dict(status='complete',records=records,
        base_only_limit_bytes=100000000,source_step=6000,optimizer_updates=0,quality_accepted=False))


if __name__=='__main__':main()
