"""Compare scalar and batched delivered coarse solvers on matched windows."""
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
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve previous batch evidence.')
    server=DATA/'jobs/live-model-review';description=json.loads((server/'job.json').read_text())
    restart=False
    try:
        if state(server)['live']:
            subprocess.run([sys.executable,'-m','distill.jobs','stop','live-model-review'],check=True,cwd=REPO)
            restart=True
            for _ in range(60):
                if not state(server)['live']:break
                time.sleep(1)
            else:raise RuntimeError('Viewer did not stop; no GPU verification performed.')
        import torch
        from distill.student import load_student
        from distill.inference import coarse_graph_forward
        from distill.benchmark_native_runtime import measure
        from distill.native_weight_quant import file_sha
        from distill.native_base import teacher_source
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
        config=json.loads((teacher_source().parent/'config.json').read_text())
        labels=torch.log(torch.tensor(config['cond_snr'])/8.)
        rows=[]
        for steps in (1,2,4):
            path=REPO/'distill/weights'/f'coarse{steps}.pt'
            model,saved=load_student(path,'cuda');model.requires_grad_(False)
            model.solver.prepare_capture(exact_kernels=False)
            checks=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for seed in (2026101491,2026101492):
                    torch.manual_seed(seed)
                    inputs=torch.randn(4,16,64,64,device='cuda')
                    inputs[:,11:]=labels.to('cuda')[None,:,None,None]
                    scalar=torch.cat([coarse_graph_forward(model,inputs[i:i+1]) for i in range(4)])
                    batched=coarse_graph_forward(model,inputs)
                    delta=(batched-scalar).float()
                    if not torch.isfinite(batched).all():raise ValueError('Nonfinite batched coarse output.')
                    checks.append(dict(seed=seed,exact=torch.equal(scalar,batched),normalized_mae=float(delta.abs().mean()),
                                       normalized_max=float(delta.abs().max())))
                # Time the same four frames per call, including graph copies.
                scalar_time=measure(lambda:[coarse_graph_forward(model,inputs[i:i+1]) for i in range(4)],20)
                batch_time=measure(lambda:coarse_graph_forward(model,inputs),20)
            row=dict(solver_steps=steps,checkpoint_sha256=file_sha(path),checkpoint_step=saved['step'],
                     checks=checks,scalar=scalar_time,batched=batch_time,
                     four_window_speedup=scalar_time['median_ms']/batch_time['median_ms'])
            rows.append(row);print(json.dumps(row),flush=True)
            del model
            torch.cuda.empty_cache()
        atomic_json(output,dict(status='complete',gpu=torch.cuda.get_device_name(),rows=rows,
            provisional=True,whole_system_idle=False,other_gpu='5090 recovery active',
            scope='Four delivered coarse64px windows; CPU geography/features/fusion excluded.',
            batch_integration_opt_in=True,code_sha256=file_sha(REPO/'distill/inference.py')))
    finally:
        if restart:
            subprocess.run([sys.executable,'-m','distill.jobs','start','live-model-review','--gpu',description['gpu'],
                            '--',*description['command']],check=True,cwd=REPO)


if __name__=='__main__':main()
