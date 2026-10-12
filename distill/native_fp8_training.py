"""TRAIN-native two-phase batches, including live student initialization states."""
from pathlib import Path

import numpy as np
import torch

from distill.native_basis_training import choose,objective,capture_blocks
from distill.native_decoded import DecodedRegion
from distill.native_feature_kd import representation_loss
from distill.native_fusion import blend,refinement_inputs,window_outputs
from distill.common import HOLDOUT_SEEDS
from distill.wet_mountains import KIND


class QATBank:
    def __init__(self):self.regions={}

    def load(self,record):
        key=(record['root'],record['file'])
        if key not in self.regions:
            region=DecodedRegion(Path(record['root'])/record['file'],'cpu')
            if region.metadata['split']!='train' or region.metadata['seed'] in HOLDOUT_SEEDS:
                raise ValueError('QAT states must come from independent TRAIN scenes.')
            self.regions[key]=region
        return self.regions[key]

    def batch(self,records,generator,step,model=None,on_policy=False):
        selected=choose(records,generator,step);windows=[];state_rows=[]
        from terrain_diffusion.inference.world_pipeline import linear_weight_window
        weight=linear_weight_window(64,'cuda',torch.float32)
        for phase,record in selected:
            region=self.load(record);stage=region.stages[phase]
            i=int(generator.integers(len(stage['inputs'])))
            inputs=stage['inputs'][i:i+1].cuda()
            if phase==1 and on_policy:
                if model is None:raise ValueError('On-policy batches require the current pupil.')
                y,x=region.ref_origins[i]
                ids=[j for j,(yy,xx) in enumerate(region.init_origins) if abs(yy-y)<=32 and abs(xx-x)<=32]
                if len(ids)!=9:raise ValueError('Complete nine-window initialization support required.')
                first=region.stages[0];origins=[region.init_origins[j] for j in ids]
                with torch.no_grad():
                    inputs0=first['inputs'][ids].cuda();labels0=first['labels'][ids].cuda()
                    velocity=model(inputs0,labels0,[first['condition'][ids].cuda()])
                    initial=blend(window_outputs(inputs0,velocity,labels0),origins,(y-32,x-32),128,weight)
                    weighted=initial[:,32:96,32:96]
                    inputs=refinement_inputs(weighted[None],region.noise[i:i+1].cuda(),stage['labels'][i:i+1].cuda())
                state_rows.append(dict(seed=record['metadata']['seed'],origin=[y,x],dependencies=9))
            windows.append(dict(inputs=inputs,labels=stage['labels'][i:i+1].cuda(),condition=stage['condition'][i:i+1].cuda()))
        batch={k:torch.cat([r[k] for r in windows]) for k in windows[0]}
        batch['mountain']=torch.tensor([r['case_kind']==KIND for _,r in selected],device='cuda')
        batch['on_policy']=bool(on_policy);batch['state_rows']=state_rows
        return batch,selected


def operator_loss(prediction,batch,target,variance,calibration):
    loss,terms=objective(prediction,batch['inputs'],batch['labels'],target,variance,calibration,batch['mountain'])
    first=batch['labels']>1.
    c0=(prediction[first,0]-target[first,0].detach()).square().mean()/variance[0]
    # Parent operator averages two phases and five channels. Adding .1*c0
    # doubles only first-phase ch0, leaving every other output supervised.
    loss=loss+.1*c0
    terms['first_channel0_operator']=c0;terms['weighted_operator']=terms['operator']+.1*c0
    return loss,terms


def kd_step(model,teacher,batch,variance,calibration,feature_weight):
    with torch.no_grad(),capture_blocks(teacher) as reference:
        target=teacher(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
                       conditional_inputs=[batch['condition'].bfloat16()]).float()
    with capture_blocks(model.net) as actual:
        prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
    feature,blocks=representation_loss(actual,reference)
    loss,terms=operator_loss(prediction,batch,target,variance,calibration)
    return loss+feature_weight*feature,terms|dict(representation=feature),blocks
