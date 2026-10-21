"""Export actual<100MB composite weights and prove exact CUDA window round trips."""
import argparse
import json
from pathlib import Path

import torch

from distill.common import atomic_json, external_path
from distill.native_onepass import noise_windows
from distill.native_onepass_residual_pack import quantized_source, export_residual_packed, ResidualPackedRuntime
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion
from distill.native_progressive_prepare import configure
from distill.native_weight_quant import file_sha


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    configure()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    manifest=export_residual_packed(saved,output,device='cuda')
    source=quantized_source(saved).cuda().eval().requires_grad_(False)
    runtime=ResidualPackedRuntime(output,'cuda')
    records,_,_=inventory();rows=[]
    for kind in ('desert-plain','wet-complex-coast','wet-high-mountain'):
        record=next(r for r in records if r['case_kind']==kind and r['coarse_variant']=='coarse4')
        region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
        for count in (1,4):
            noises=noise_windows(record['metadata']['seed'],region.ref_origins[:count],'cuda')
            condition=region.stages[1]['condition'][:count]
            expected=source.windows(noises,condition)
            actual=runtime(noises,condition)
            if not torch.equal(actual,expected):
                raise ValueError(f'Packed one-pass changed windows:{kind}:{count}:max{float((actual-expected).abs().max())}')
            rows.append(dict(kind=kind,batch=count,exact=True))
    atomic_json(output/'audit.json',dict(status='complete',source_sha256=file_sha(args.checkpoint),
        weight_bytes=manifest['onepass_wrapper']['download_weight_bytes'],checks=rows,gpu=torch.cuda.get_device_name(),
        qat_completed=saved.get('architecture')=='native-two-noise-onepass-residual-nf4',quality_accepted=False,
        note='Exactfake-NF4/packed outputs; PTQ fidelity versus original float andteacher still needs separate evaluation.'))
    print(json.dumps(dict(status='complete',bytes=manifest['onepass_wrapper']['download_weight_bytes'],exact_controls=len(rows))),flush=True)


if __name__=='__main__':main()
