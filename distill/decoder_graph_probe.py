"""Exact decoder capture controls and provisional eager/graph window timings.

The selected viewer job is stopped forGPU verification and restored in finally.
Recovery on5090 stays active; measurements are provisional until a full-chain
idle run. Run under the shared4090 lock, between one-pass pilot segments.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path
from distill.jobs import state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=REPO/'distill/weights/decoder.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--onepass-checkpoint',type=Path,
        default=DATA/'ckpt/native-onepass160x2-v1/pilot/step000500.pt')
    args=parser.parse_args()
    output=external_path(args.output)
    if output.exists():raise ValueError('Preserve prior capture evidence.')
    server=DATA/'jobs/live-model-review'
    description=json.loads((server/'job.json').read_text())
    was_live=state(server)['live']
    stopped_server=False
    try:
        if was_live:
            subprocess.run([sys.executable,'-m','distill.jobs','stop','live-model-review'],check=True,cwd=REPO)
            stopped_server=True
            for _ in range(60):
                if not state(server)['live']:break
                time.sleep(1)
            else:raise RuntimeError('Viewer did not stop; no GPU verification performed.')
        import torch
        from distill.student import load_student
        from distill.inference import student_graph_forward
        from distill.benchmark_native_runtime import measure
        from distill.native_weight_quant import file_sha
        model,saved=load_student(args.checkpoint,'cuda')
        if model.config.stage!='decoder':raise ValueError('A delivered decoder is required.')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
        checks=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for seed in (2026101461,2026101462,2026101463):
                torch.manual_seed(seed)
                inputs=torch.randn(1,16,512,512,device='cuda')
                expected=model(inputs)
                actual=student_graph_forward(model,inputs,stage='decoder')
                if not torch.equal(expected,actual):raise ValueError('Decoder graph changes native window output.')
                checks.append(dict(seed=seed,exact=True,finite=bool(torch.isfinite(actual).all())))
                if not checks[-1]['finite']:raise ValueError('Nonfinite decoder graph output.')
            # Changing weights must invalidate the cached graph and its constants.
            parameter=next(model.parameters())
            parameter.add_(1e-5)
            expected=model(inputs)
            actual=student_graph_forward(model,inputs,stage='decoder')
            if not torch.equal(expected,actual):raise ValueError('Weight-version graph invalidation failed.')
            # Reload delivered weights before timing.
            model.load_state_dict(saved['ema'] if 'ema' in saved else saved['model'])
            eager=measure(lambda:model(inputs),20)
            graph=measure(lambda:student_graph_forward(model,inputs,stage='decoder'),20)
        onepass_saved=torch.load(args.onepass_checkpoint,map_location='cpu',weights_only=True,mmap=True)
        from distill.native_onepass_runtime import OnePassRuntime
        from distill.native_onepass_graph import OnePassGraph
        from distill.native_onepass import noise_windows
        runtime=OnePassRuntime(onepass_saved,'cuda').requires_grad_(False)
        captured=OnePassGraph(runtime)
        onepass_checks=[]
        with torch.no_grad():
            for batch in (1,4,16):
                for seed in (615001,615002):
                    noises=noise_windows(seed,[(0,32*i) for i in range(batch)],'cuda')
                    condition=torch.randn(batch,58,device='cuda')
                    expected=runtime(noises,condition)
                    actual=captured(noises,condition)
                    if not torch.equal(expected,actual):raise ValueError('One-pass captured windows differ.')
                    onepass_checks.append(dict(batch=batch,seed=seed,exact=True))
        atomic_json(output,dict(status='complete',gpu=torch.cuda.get_device_name(),checkpoint_sha256=file_sha(args.checkpoint),
            checks=checks,changed_weight_recapture_exact=True,eager=eager,graph=graph,
            window_speedup=eager['median_ms']/graph['median_ms'],provisional=True,whole_system_idle=False,
            scope='Delivered decoder512px windows, includes graph input/output copies; CPU feature construction and fusion excluded.',
            viewer_stopped_for_verification=was_live,other_gpu='5090 recovery active',
            onepass_checkpoint_sha256=file_sha(args.onepass_checkpoint),onepass_checks=onepass_checks,
            code_sha256=file_sha(REPO/'distill/inference.py')))
        print(json.dumps(dict(status='complete',window_speedup=eager['median_ms']/graph['median_ms'],exact_checks=len(checks))),flush=True)
    finally:
        if stopped_server:
            subprocess.run([sys.executable,'-m','distill.jobs','start','live-model-review','--gpu',description['gpu'],
                            '--',*description['command']],check=True,cwd=REPO)


if __name__=='__main__':main()
