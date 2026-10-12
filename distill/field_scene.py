"""Native128 teacher captures as two-phase256 field supervision, no requalification."""
import json

import numpy as np
import torch

from distill.native_fusion import refinement_inputs,window_outputs


class FieldScene:
    def __init__(self,path,device='cpu',shift=(0,0)):
        from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
        with np.load(path,allow_pickle=False) as sample:
            self.metadata=json.loads(str(sample['metadata']))
            labels,contexts,condition=sample['labels'],sample['contexts'],sample['condition']
            first,second=float(labels.max()),float(labels.min())
            first_ids=np.flatnonzero(labels==first);second_ids=np.flatnonzero(labels==second)
            ys=sorted(set(int(v)*32 for v in contexts[first_ids,1]));xs=sorted(set(int(v)*32 for v in contexts[first_ids,2]))
            if len(ys)!=7 or len(xs)!=7 or len(first_ids)!=49 or len(second_ids)!=25:
                raise ValueError('Complete native128 teacher capture with49/25 windows required.')
            lookup={(int(contexts[i,1])*32,int(contexts[i,2])*32):int(i) for i in first_ids}
            grid=np.stack([condition[lookup[(y,x)]] for y in ys for x in xs]).reshape(7,7,58).transpose(2,0,1)
            for i in second_ids:
                key=(int(contexts[i,1])*32,int(contexts[i,2])*32)
                if not np.array_equal(condition[i],condition[lookup[key]]):
                    raise ValueError('Both phases must use the same captured coarse condition at each location.')
            weighted=sample['initial_weighted'].copy()
            if weighted.shape!=(6,192,192) or not (weighted[-1]>0).all():
                raise ValueError('Teacher initial192 field needs complete positive coverage.')
            self.initial=torch.from_numpy(weighted[:-1]/weighted[-1:]).to(device)
            self.target=torch.from_numpy(sample['final_latents'].copy()).to(device)
            self.mask=torch.from_numpy(sample['final_mask'].copy()).to(device)
            centre=(self.metadata['y']+32,self.metadata['x']+32)
            selected=[]
            for ids in (first_ids,second_ids):
                matches=[int(i) for i in ids if (int(contexts[i,1])*32,int(contexts[i,2])*32)==centre]
                if len(matches)!=1:raise ValueError('Central teacher KD window missing.')
                selected.append(matches[0])
            self.teacher_windows={k:torch.from_numpy(sample[k][selected].copy()).to(device)
                                  for k in ('inputs','labels','condition','target')}
        if self.target.shape!=(5,128,128) or not self.mask.all():
            raise ValueError('Complete original blended128 final target required.')
        y,x=self.metadata['y'],self.metadata['x']
        if (ys[0],xs[0])!=(y-64,x-64):raise ValueError('Native field conditioning alignment changed.')
        self.origin=(y-64+shift[0],x-64+shift[1])
        self.centres=(ys[0]+32-self.origin[0],xs[0]+32-self.origin[1])
        self.condition=torch.from_numpy(grid.copy()).to(device)
        self.labels=torch.tensor([first,second],device=device)
        self.dtype=torch.bfloat16 if torch.device(device).type=='cuda' else torch.float32
        self.noise=[torch.from_numpy(gaussian_noise_patch(self.metadata['seed']+5819+phase,
            *self.origin,256,256,channels=5,tile_h=64,tile_w=64)).to(device) for phase in (0,1)]
        zeros=torch.zeros((1,6,256,256),device=device);zeros[:,-1]=1
        self.inputs0=refinement_inputs(zeros,self.noise[0][None],self.labels[:1],self.dtype)


def field_scene_forward(model,scene,checkpointed=False,features=False,initial_override=None):
    from torch.utils.checkpoint import checkpoint
    condition=scene.condition[None];origins=[scene.origin];centres=[scene.centres]
    def evaluate(inputs,labels):
        return model(inputs,labels,condition,origins,centres,return_features=features)
    initial_heads=None
    if hasattr(model.net,'field_window_heads'):
        from distill.field_heads import combine_heads
        def evaluate_heads(inputs,labels):
            return model(inputs,labels,condition,origins,centres,return_features=features,return_head_velocities=True)
        first=(checkpoint(evaluate_heads,scene.inputs0,scene.labels[:1],use_reentrant=False)
               if checkpointed else evaluate_heads(scene.inputs0,scene.labels[:1]))
        velocity0,features0,head_velocities=first
        b,_,c,h,w=head_velocities.shape
        initial_heads=window_outputs(scene.inputs0.repeat_interleave(4,dim=0),head_velocities.reshape(b*4,c,h,w),
            scene.labels[:1].expand(b*4),scene.dtype).reshape(b,4,c,h,w)
        initial=combine_heads(initial_heads,origins)
    else:
        first=(checkpoint(evaluate,scene.inputs0,scene.labels[:1],use_reentrant=False)
                   if checkpointed else evaluate(scene.inputs0,scene.labels[:1]))
        velocity0,features0=first if features else (first,None)
        initial=window_outputs(scene.inputs0,velocity0,scene.labels[:1],scene.dtype)
    refinement=initial if initial_override is None else initial_override.detach()[None]
    weighted=torch.cat((refinement,torch.ones_like(refinement[:,:1])),dim=1)
    inputs1=refinement_inputs(weighted,scene.noise[1][None],scene.labels[1:],scene.dtype)
    second=(checkpoint(evaluate,inputs1,scene.labels[1:],use_reentrant=False)
               if checkpointed else evaluate(inputs1,scene.labels[1:]))
    velocity1,features1=second if features else (second,None)
    final=window_outputs(inputs1,velocity1,scene.labels[1:],scene.dtype)
    return dict(initial=initial,final=final,initial192=initial[:,:,32:224,32:224],
                core=final[:,:,64:192,64:192],inputs1=inputs1,velocity0=velocity0,velocity1=velocity1,
                features0=features0,features1=features1,initial_heads=initial_heads)
