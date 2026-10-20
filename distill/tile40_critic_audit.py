"""Measure critic separation and actual generator gradient balance on fixed cases.

This diagnoses a training loss;it does not certify complete terrain morphology.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import atomic_json, external_path
from distill.student import load_student
from distill.tile40_adversarial_pilot import TerrainCritic
from distill.tile40_decoder_pilot import References
from distill.tile40_encoder_pilot import terms


def norm(gradients):
    return float(torch.sqrt(sum(g.double().square().sum() for g in gradients if g is not None)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--decoder',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=external_path(args.output)
    if output.exists():
        raise ValueError('Use a fresh audit output.')
    torch.set_num_threads(8)
    model,saved=load_student(args.checkpoint,ema=False)
    decoder,_=load_student(args.decoder)
    model.requires_grad_(False);decoder.requires_grad_(False)
    critic=TerrainCritic().eval()
    critic.load_state_dict(saved['critic']);critic.requires_grad_(False)
    variance=saved['critic_std'].flatten().square()
    refs=References(args.dataset)
    rows=[]
    for split in ('train','val'):
        selected=sorted(refs.splits[split],key=lambda r:(r['case']['seed'],r['case']['climate']))[:30]
        for index,row in enumerate(selected):
            capture={}
            with torch.no_grad():
                loss,metrics=terms(model,decoder,decoder,row,(index%2)*96,((index//2)%2)*96,
                    variance,'cpu',1.,.01,capture,.1)
                real=float(critic(capture['height_target'],capture['terrain_conditioning']).mean())
                fake=float(critic(capture['height_prediction'],capture['terrain_conditioning']).mean())
            rows.append(dict(split=split,case=row['case'],real_score=real,fake_score=fake,
                             physical_loss=float(loss),height_mae_m=float(metrics['height_mae_m'])))
    training=[r for r in rows if r['split']=='train']
    scores=sorted([r[k] for r in training for k in ('real_score','fake_score')])
    candidates=[scores[0]-1.]+[(a+b)/2 for a,b in zip(scores,scores[1:])]+[scores[-1]+1.]
    accuracy=lambda items,t:sum((r['real_score']>t)+(r['fake_score']<=t) for r in items)/(2*len(items))
    threshold=max(candidates,key=lambda t:accuracy(training,t))
    groups={}
    for split in ('train','val'):
        items=[r for r in rows if r['split']==split]
        groups[split]=dict(accuracy_at_train_threshold=accuracy(items,threshold),
            pair_order_accuracy=float(np.mean([r['real_score']>r['fake_score'] for r in items])),
            auc=float(np.mean([[a['real_score']>b['fake_score'] for b in items] for a in items])),
            mean_real=float(np.mean([r['real_score'] for r in items])),mean_fake=float(np.mean([r['fake_score'] for r in items])))
    gradients=[]
    model.requires_grad_(True)
    parameters=tuple(model.parameters())
    for relief in ('mountains','hills','plain'):
        row=min((r for r in refs.splits['train'] if r['case']['relief']==relief),key=lambda r:r['case']['seed'])
        capture={}
        loss,_=terms(model,decoder,decoder,row,0,0,variance,'cpu',1.,.01,capture,.1)
        adversarial=-critic(capture['height_prediction'],capture['terrain_conditioning']).mean()*.3
        physical_grad=norm(torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True))
        critic_grad=norm(torch.autograd.grad(adversarial,parameters,allow_unused=True))
        gradients.append(dict(relief=relief,physical_norm=physical_grad,weighted_adversarial_norm=critic_grad,
                              weighted_ratio=critic_grad/max(physical_grad,1e-12)))
    atomic_json(output,dict(status='complete',generator_weights='raw',step=saved['step'],rows=rows,
        train_selected_threshold=threshold,groups=groups,gradients=gradients,
        scope='CPU FP32 training-proxy critic/parameter-gradient audit;not complete40km quality or browser timing.'))
    print(json.dumps(dict(groups=groups,gradients=gradients)),flush=True)


if __name__=='__main__':
    main()
