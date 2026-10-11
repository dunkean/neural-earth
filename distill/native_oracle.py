"""Channel-swap diagnostics between complete teacher and native-student fields.

Oracle outputs are normalized 64-square fields with non-overlapping storage.
They are diagnostic hybrids, never model choices or exported candidates.
"""
from __future__ import annotations

import torch


def mix_channels(student,teacher,mode):
    if student.shape != teacher.shape or student.shape[0] != 6:
        raise ValueError('Matching weighted five-channel native fields required.')
    if torch.any(student[-1:] <= 0) or torch.any(teacher[-1:] <= 0):
        raise ValueError('Oracle fields require complete coverage.')
    a,b=student[:-1]/student[-1:],teacher[:-1]/teacher[-1:]
    if mode=='teacher_height':
        values=torch.cat((a[:4],b[4:5]))
    elif mode=='teacher_features':
        values=torch.cat((b[:4],a[4:5]))
    elif mode=='teacher_all':
        values=b
    else:
        raise ValueError('Unknown native channel oracle.')
    return torch.cat((values,torch.ones_like(student[-1:])))


def mixed_tensor(student,teacher,mode,store):
    from infinite_tensor import InfiniteTensor,TensorWindow
    window=TensorWindow(size=(6,64,64),stride=(6,64,64))
    def forward(contexts,candidates,references):
        return [mix_channels(a,b,mode) for a,b in zip(candidates,references)]
    return InfiniteTensor(shape=(6,None,None),f=forward,output_window=window,
        args=(student,teacher),args_windows=(window,window),batch_size=16,
        device=student.device,dtype=torch.float32,tile_store=store,tensor_id='native_oracle_'+mode)
