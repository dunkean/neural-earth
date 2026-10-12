"""Read scalar gains and their effective BF16 values; no models or CUDA jobs."""
import argparse
import csv
import json
from pathlib import Path

import torch
from safetensors import safe_open

from distill.common import DATA,atomic_json,external_path
from distill.native_base import teacher_source


def scalar_values(state):
    rows={}
    for name,value in state.items():
        if value.numel()!=1:continue
        effective=value.detach().reshape(()).bfloat16()
        rows[name]=dict(stored=float(value),stored_dtype=str(value.dtype),
            bf16=float(effective),bf16_bits=int(effective.view(torch.int16))&65535)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve previous scalar evidence.')
    torch.set_num_threads(2)
    checkpoints=[]
    root=DATA/'ckpt/native-weight-nf4-100k'
    for p in sorted(root.glob('step??????.pt')):checkpoints.append((p,'ema','nf4long'))
    for p in sorted((root/'raw').glob('step??????.pt')):checkpoints.append((p,'ema','nf4longraw'))
    for arm in ('a','b'):
        r=DATA/'ckpt/native-nf4-objective-ablation'/arm
        for p in sorted(r.glob('step??????.pt')):checkpoints.append((p,'ema','nf4abl'+arm))
        for p in sorted((r/'raw').glob('step??????.pt')):checkpoints.append((p,'ema','nf4abl'+arm+'raw'))
        for state in ('model','ema'):checkpoints.append((r/'latest.pt',state,'nf4abl'+arm+('raw' if state=='model' else '')))
    for state in ('model','ema'):checkpoints.append((root/'latest.pt',state,'nf4longraw' if state=='model' else 'nf4long'))
    rows=[]
    for p,state,label in checkpoints:
        saved=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        scalars=scalar_values(saved[state]);step=saved['step']
        plate=DATA/'_base'/f'native-weight-{label}-coarse4-{step:06d}.json'
        bias=[]
        if plate.exists():
            report=json.loads(plate.read_text())
            bias=[dict(kind=r['kind'],seed=r['seed'],lod=r['lod'],bias_m=r['height_bias_m'],mae_m=r['mae_m']) for r in report['rows']]
        stat=p.stat()
        rows.append(dict(path=str(p),state=state,label=label,step=step,
            source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,scalars=scalars,bias=bias,
            precision_note='FP32 masters unavailable in BF16 snapshots; do not reconstruct their values.'))
        del saved
    with safe_open(teacher_source()/'diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as handle:
        teacher=scalar_values({'net.'+n:handle.get_tensor(n) for n in handle.keys() if n=='out_gain' or n.endswith('emb_gain')})
    pairs=[]
    for step in (42500,45000,47500,50000):
        for raw in (False,True):
            suffix='raw' if raw else ''
            a=next(r for r in rows if r['label']=='nf4abla'+suffix and r['step']==step and r['state']=='ema')
            b=next(r for r in rows if r['label']=='nf4ablb'+suffix and r['step']==step and r['state']=='ema')
            changed=[name for name in a['scalars'] if a['scalars'][name]['bf16_bits']!=b['scalars'][name]['bf16_bits']]
            pairs.append(dict(step=step,raw=raw,changed_scalars=changed,
                out_gain_a=a['scalars']['net.out_gain'],out_gain_b=b['scalars']['net.out_gain']))
    atomic_json(output,dict(status='complete',teacher=teacher,rows=rows,pairs=pairs,
        weight_updates=0,cuda_initialized=torch.cuda.is_initialized(),
        scope='Scalar values and matching scene biases; correlation alone is not causal evidence.'))
    with output.with_suffix('.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['label','step','state','scalar','stored','stored_dtype','bf16','bf16_bits']);writer.writeheader()
        for row in rows:
            for name,value in row['scalars'].items():writer.writerow(dict(label=row['label'],step=row['step'],state=row['state'],scalar=name,**value))
    print(json.dumps(dict(checkpoints=len(rows),teacher_out_gain=teacher['net.out_gain'],pairs=pairs)),flush=True)


if __name__=='__main__':main()
