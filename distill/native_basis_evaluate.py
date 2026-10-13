"""Independent4090 checkpoint evaluator and TRAIN-only autonomous-state producer."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.jobs import state
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_lowrank import NativeLowrankBase
from distill.native_multilod_train import validate
from distill.native_multilod_eval import validation_records
from distill.native_mixed_bank import KINDS, VARIANTS
from distill.wet_mountains import KIND, digest


@torch.no_grad()
def rollouts(model,train,weight,output,step,sha):
    destination=output/f'rollouts-{step:06d}'; rows=[]
    rng=np.random.default_rng(18210+step)
    for variant in VARIANTS:
        for kind in (None,*KINDS,KIND):
            pool=[r for r in train if r['coarse_variant']==variant and r['case_kind']==kind]
            selected=[]; used=set()
            for index in rng.permutation(len(pool)):
                row=pool[index]
                if row['metadata']['seed'] in used:continue
                selected.append(row);used.add(row['metadata']['seed'])
                if len(selected)==2:break
            for index,record in enumerate(selected):
                region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
                result=decoded_region_forward(model,region,weight)
                path=destination/f'{variant}-{kind or "general"}-{index}.npz'
                arrays=dict(inputs=result['inputs1'].float().cpu().numpy(),
                    labels=region.stages[1]['labels'].float().cpu().numpy(),
                    condition=region.stages[1]['condition'].float().cpu().numpy(),
                    metadata=np.array(json.dumps(dict(split='train',seed=record['metadata']['seed'],
                        checkpoint_sha256=sha,coarse_variant=variant,case_kind=kind,step=step))))
                if any(not np.isfinite(v).all() for k,v in arrays.items() if k!='metadata'):
                    raise FloatingPointError('Nonfinite autonomous state.')
                atomic_write(path,lambda f:np.savez(f,**arrays))
                rows.append(dict(file=str(path),seed=record['metadata']['seed'],coarse_variant=variant,
                                 case_kind=kind,record=record))
    atomic_json(output/'rollouts-index.json',dict(step=step,checkpoint_sha256=sha,split='train',rows=rows))


def mountain_checks(output,label,step):
    rows=[]
    for variant in VARIANTS:
        name=label if variant=='coarse4' else label+'-'+variant
        data=json.loads((DATA/'_base'/f'mountains-{name}-{step:06d}.json').read_text())
        for row in data['rows']:
            if row['lod']==0:
                tpi=next(m for m in row['shape']['topographic_position'] if m['sigma_pixels']==8.)
                rows.append(dict(seed=row['seed'],coarse=variant,mae_m=row['mae_m'],**tpi))
    return rows


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DATA/'ckpt/native-basis-decoder-quarter-pilot')
    parser.add_argument('--trainer-job',default='base-basis-quarter-train')
    parser.add_argument('--label',default='basisquarter')
    args=parser.parse_args()
    output=external_path(args.run)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train,root,manifest=inventory()
    stats=json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())
    records=validation_records(manifest,stats)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32)
    decoder=OriginalDecoder('cuda');variance=torch.tensor(stats['variance'],device='cuda')
    from distill.wet_mountain_progress import publish
    done=set();post_cut=None;best=float('inf');started=time.monotonic()
    try:
        while True:
            candidates=sorted(output.glob('step[0-9][0-9][0-9][0-9][0-9][0-9].pt'))
            for path in candidates:
                if path.name in done:continue
                sha=digest(path);saved=torch.load(path,map_location='cpu',weights_only=True)
                model=NativeLowrankBase(saved['config'],saved['factor_ranks']).cuda().eval().requires_grad_(False)
                model.load_state_dict(saved['ema']);step=saved['step']
                details=validate(model,records,root,weight,decoder,variance,True)
                subprocess.run([sys.executable,'-m','distill.native_progress_images',
                    '--checkpoints',str(path),'--label',args.label],cwd=REPO,check=True)
                for coarse in ('original','coarse2'):
                    publish(path,args.label+'-'+coarse,coarse=coarse)
                mountains=mountain_checks(output,args.label,step)
                # Full-rank control errors are numerical rounding, not the
                # physical BF16/FP32 tolerance. Don't impose a4m gate from it.
                mountain_pass=all(r['tpi_correlation']>=.98 and r['valley_retention']>=.96 and
                    .93<=r['tpi_rms_ratio']<=1.07 and r['mae_m']<=35. for r in mountains)
                other_pass=True
                for group in details['groups'].values():
                    for scale,morph in group['morphology'].items():
                        if morph is None:continue
                        if scale=='fine':other_pass &= morph['tpi_correlation']>=.97 and .95<=morph['tpi_rms_ratio']<=1.08
                        else:other_pass &= morph['tpi_correlation']>=.80 and .85<=morph['tpi_rms_ratio']<=1.15
                reasons=[]
                if step==1000:post_cut=details['feature_relative_mse']
                if step>=1500:
                    if post_cut is not None and details['feature_relative_mse']>2*post_cut:
                        reasons.append('latent feature error doubled since the post-cut1000 checkpoint')
                    if any(r['tpi_correlation']<.75 or r['valley_retention']<.4 for r in mountains):
                        reasons.append('catastrophic wet-mountain valley loss after the anneal grace period')
                diagnostic_score=details['selection_score']+float(np.mean([1-r['valley_iou'] for r in mountains]))
                stage_pass=step>=2000 and mountain_pass and other_pass and details['feature_relative_mse']<=.02
                row=dict(step=step,checkpoint_sha256=sha,mountains=mountains,
                    stage_pass=stage_pass,hard_stop=bool(reasons),reasons=reasons,
                    feature_relative_mse=details['feature_relative_mse'],selection_score=diagnostic_score,
                    accepted=False,note='Interior stage qualification only; application, cold seams, final worlds and trained idle speed still pending.')
                atomic_json(output/f'eval-{step:06d}.json',row|dict(details=details))
                append_json(output/'evaluation.jsonl',row)
                if step>=1000 and not reasons and diagnostic_score<best:
                    best=diagnostic_score
                    atomic_json(output/'best-diagnostic.json',dict(checkpoint=str(path),step=step,
                        checkpoint_sha256=sha,stage_pass=stage_pass,accepted=False))
                if step>=1000 and not reasons:rollouts(model,train,weight,output,step,sha)
                atomic_json(output/'evaluation-status.json',row|dict(status='complete'))
                print(json.dumps(row),flush=True)
                done.add(path.name);del model,saved;gc.collect();torch.cuda.empty_cache()
            owner=state(DATA/'jobs'/args.trainer_job)
            if not owner['live']:
                if owner['status'] in ('complete','failed'):
                    # Recheck atomic checkpoint inventory once before closing.
                    remaining={p.name for p in output.glob('step[0-9][0-9][0-9][0-9][0-9][0-9].pt')}-done
                    if not remaining:break
                elif time.monotonic()-started>600:
                    raise RuntimeError('Verified trainer handle did not appear.')
            time.sleep(30)
    except Exception as exc:
        atomic_json(output/'evaluation-status.json',dict(status='failed',error=str(exc),hard_stop=True,accepted=False))
        raise


if __name__=='__main__':main()
