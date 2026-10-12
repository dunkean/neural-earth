"""Teacher-width field numerical control and actual two-phase gradient smoke."""
import argparse
import json
import time

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path
from distill.native_base import teacher_source
from distill.field_base import FieldBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.wet_mountains import digest
from terrain_diffusion.models.edm_unet import EDMUnet2D


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(DATA/'eval/field-base-initial-probe'))
    parser.add_argument('--gradients',action='store_true')
    args=parser.parse_args();root=external_path(args.output)
    torch.set_num_threads(8);torch.manual_seed(128401)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().eval().requires_grad_(False)
    model=FieldBase(dict(teacher.config)).cuda().eval();model.net.load_state_dict(teacher.state_dict())
    manifest=json.loads((DATA/'crops/native-mixed-scenes-v2/manifest.json').read_text())
    row=next(r for r in manifest['scenes'] if r['metadata']['split']=='train' and r['coarse_variant']=='coarse4')
    path=DATA/'crops/native-mixed-scenes-v2'/row['file']
    with np.load(path) as sample:
        inputs=torch.from_numpy(sample['inputs'][[0,-1]].copy()).cuda()
        labels=torch.from_numpy(sample['labels'][[0,-1]].copy()).cuda()
        conditions=torch.from_numpy(sample['condition'][[0,-1]].copy()).cuda()
    with torch.no_grad():
        reference=teacher(inputs.float(),noise_labels=labels.float(),conditional_inputs=[conditions.float()])
        actual=model.net(inputs.float(),noise_labels=labels.float(),conditions=conditions[:,:,None,None],
                         origins=[(0,0),(64,64)],centres=[(32,32),(32,32)])
        error=actual-reference
        control=dict(relative_mse=float(error.square().mean()/reference.square().mean()),max_abs=float(error.abs().max()))
        if control['relative_mse']>1e-8 or control['max_abs']>.002:
            raise ValueError(f'Field64 constant-condition control failed: {control}')
    del teacher;torch.cuda.empty_cache()
    snapshot=dict(config=model.config,ema=model.state_dict(),architecture='native-field',step=0,
        teacher_revision=REVISION,teacher_weights_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        phases=2,field_size=256,core_size=128,halo=64,condition_grid_spacing=32,
        attention='world-aligned64 local windows',evaluation_only=True,accepted=False)
    atomic_write(root/'initial.pt',lambda f:torch.save(snapshot,f))
    scene=FieldScene(path,'cuda');started=time.monotonic()
    model.requires_grad_(args.gradients)
    with torch.set_grad_enabled(args.gradients):
        result=field_scene_forward(model,scene,checkpointed=args.gradients)
        variance=torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'],device='cuda')[None,:,None,None]
        initial=((result['initial192']-scene.initial[None]).square()/variance).mean()
        final=((result['core']-scene.target[None]).square()/variance).mean()
        loss=initial+final
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite field forward.')
        if args.gradients:
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):
                raise ValueError('No field pupil gradients.')
        else:norm=0.
    atomic_json(root/'probe.json',dict(status='complete',gpu=torch.cuda.get_device_name(),
        control=control,initial_relative_mse=float(initial.detach()),final_relative_mse=float(final.detach()),
        gradients=args.gradients,grad_norm=float(norm),peak_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),
        sources_sha256={p:digest(REPO/'distill'/p) for p in ('field_base.py','field_scene.py','field_probe.py')},
        accepted=False,note='Untrained field geometry and real gradient viability; not terrain acceptance or a halo/seam proof.'))
    print(json.dumps(dict(initial=float(initial.detach()),final=float(final.detach()),seconds=time.monotonic()-started,
                         peak_bytes=torch.cuda.max_memory_allocated())),flush=True)


if __name__=='__main__':main()
