"""Provisional matched T2 versus true one-evaluation windows on the same GPU.

Reports graph/eager timings atbatch1/4/16. It excludes global between-pass
fusion, coarse, decoder and CPU requests; full-chain acceptance is separate.
"""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from distill.common import atomic_json, external_path
from distill.native_base import teacher_source
from distill.native_onepass import noise_windows
from distill.native_onepass_runtime import OnePassRuntime
from distill.native_phase_loss import PHASE_LABELS
from distill.native_fusion import window_outputs, refinement_inputs
from distill.native_weight_quant import file_sha
from distill.benchmark_native_runtime import measure
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_inference import prepare_models


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=20)
    args=parser.parse_args()
    output=external_path(args.output)
    if output.exists():raise ValueError('Preserve earlier timings.')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    student=OnePassRuntime(saved,'cuda').eval().requires_grad_(False)
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    from terrain_nn_constants import prepare_constants
    prepare_constants(SimpleNamespace(device='cuda',coarse_model=teacher,base_model=student.net,
                                      decoder_model=teacher),exact_kernels=False)
    rows=[]
    for batch in (1,4,16):
        noises=noise_windows(615123,[(0,32*i) for i in range(batch)],'cuda').bfloat16()
        condition=torch.randn(batch,58,device='cuda',dtype=torch.bfloat16)
        first_label=torch.full((batch,),PHASE_LABELS[0],device='cuda',dtype=torch.bfloat16)
        second_label=torch.full((batch,),PHASE_LABELS[1],device='cuda',dtype=torch.bfloat16)
        first=torch.sin(first_label[:,None,None,None])*noises[:,:5]
        def reference():
            velocity=teacher(first,noise_labels=first_label,conditional_inputs=[condition])
            initial=window_outputs(first,velocity,first_label)
            t=second_label[:,None,None,None]
            # Unit coverage is known here; avoid the production helper's host
            # assertion inside capture. Arithmetic retains its BF16 order.
            second=(torch.cos(t)*(initial.bfloat16()*.5)+torch.sin(t)*(noises[:,5:]*.5))/.5
            velocity=teacher(second,noise_labels=second_label,conditional_inputs=[condition])
            return window_outputs(second,velocity,second_label)
        def candidate():return student(noises,condition)
        for variant,fn in (('teacherT2',reference),('studentT1',candidate)):
            eager=measure(fn,args.iterations)
            for _ in range(3):fn()
            torch.cuda.synchronize()
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(3):fn()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=fn()
            captured=measure(graph.replay,args.iterations)
            if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite timed output.')
            row=dict(batch=batch,variant=variant,eager=eager,graph=captured)
            rows.append(row);print(json.dumps(row),flush=True)
            del graph,result;gc.collect();torch.cuda.empty_cache()
    gains=[]
    for batch in (1,4,16):
        a,b=[next(r for r in rows if r['batch']==batch and r['variant']==variant) for variant in ('teacherT2','studentT1')]
        gains.append(dict(batch=batch,eager_speedup=a['eager']['median_ms']/b['eager']['median_ms'],
                          graph_speedup=a['graph']['median_ms']/b['graph']['median_ms']))
    atomic_json(output,dict(status='complete',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        checkpoint_sha256=file_sha(args.checkpoint),checkpoint_step=saved['step'],rows=rows,gains=gains,
        solver_evaluations=dict(teacher=2,student=1),exact_kernels=False,
        whole_system_idle=False,provisional=True,quality_accepted=False,
        scope='Network windows with local solver arithmetic; global T2 overlap fusion/coarse/decoder/CPU are excluded.',
        note='4090 shares Windows display/live viewer;5090 trains recovery. Final measurements require idle deployment protocol.'))


if __name__=='__main__':main()
