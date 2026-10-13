"""Check every reconstructed compact tensor against the QAT's effective weights."""
import argparse
import json
import math
from pathlib import Path

import torch

from distill.common import atomic_json, external_path
from distill.native_weight_quant import encode, file_sha, load_packed
from terrain_diffusion.models.mp_layers import normalize


@torch.no_grad()
def audit(saved,packed,device='cuda'):
    packed=Path(packed);header=json.loads((packed/'manifest.json').read_text())
    if saved.get('architecture')!='native-weight-quant':raise ValueError('Native weight-QAT source required.')
    for key in ('scheme','block','step','teacher_revision','teacher_weights_sha256','quantized_layers'):
        if saved[key]!=header[key]:raise ValueError('Source/packed metadata mismatch: '+key)
    if json.dumps(saved['config'],sort_keys=True)!=json.dumps(header['config'],sort_keys=True):
        raise ValueError('Source/packed configuration mismatch.')
    if torch.device(device).type!=torch.device(header['quantization_device']).type:
        raise ValueError('Audit on the export device type; CPU/CUDA normalization reductions can differ.')
    reconstructed=load_packed(packed,device)
    target=reconstructed.state_dict();source=saved['ema']
    if set(source)!=set(target):raise ValueError('Source/packed tensor inventory mismatch.')
    quantized={'net.'+name+'.weight' for name in saved['quantized_layers']}
    rows=[]
    for name,value in source.items():
        actual=target[name].bfloat16()
        expected=value.to(device=device,dtype=torch.bfloat16)
        if name in quantized:
            effective=normalize(expected.float())/math.sqrt(expected[0].numel())
            expected=encode(effective,saved['scheme'],saved['block'])[2].bfloat16()
        exact=torch.equal(expected,actual)
        row=dict(tensor=name,quantized=name in quantized,exact=exact,
                 max_abs=0. if exact else float((actual.float()-expected.float()).abs().max()))
        rows.append(row)
    return dict(status='complete' if all(r['exact'] for r in rows) else 'failed',rows=rows,
        tensor_count=len(rows),quantized_kernel_count=len(quantized),
        stored_bytes=header['bytes'],manifest_sha256=file_sha(packed/'manifest.json'),
        weights_sha256=header['sha256'],device=device,
        gpu=torch.cuda.get_device_name() if torch.device(device).type=='cuda' else None,
        scope='All BF16 effective kernel/parameter values; forward controls and physical quality remain separate.',
        accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--packed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(8)
    output=external_path(args.output)
    if output.exists():raise ValueError('Keep previous audit evidence; choose a new output.')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    report=audit(saved,args.packed,args.device)
    report.update(checkpoint=str(args.checkpoint),checkpoint_sha256=file_sha(args.checkpoint),
                  source_sha256=file_sha(Path(__file__)))
    atomic_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if report['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
