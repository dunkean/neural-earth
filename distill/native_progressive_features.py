"""Feature supervision for a physically compacted student.

The student's channels are consecutive after compaction. The teacher retains
its original coordinates, so only the teacher is indexed by the saved mapping.
This is separate from the first-stage loss, which indexes both full-width nets.
"""
import torch


def compact_feature_loss(actual, reference, original_teacher_mapping):
    terms = []
    for name, value in actual.items():
        weight = name+'.weight' if name.endswith('_conv') else name+'.conv_res1.weight'
        coordinates = original_teacher_mapping[weight]['output']
        target = reference[name].detach().float()
        if coordinates is None or len(coordinates) != value.shape[1]:
            raise ValueError('Compacted feature mapping has the wrong width: '+name)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError('Compacted feature mapping repeats a teacher channel: '+name)
        indices = torch.tensor(coordinates, device=target.device, dtype=torch.long)
        target = target.index_select(1, indices)
        if target.shape != value.shape:
            raise ValueError('Student and selected teacher feature shapes differ: '+name)
        energy = target.square().mean((0, 2, 3), keepdim=True).clamp_min(1e-4)
        terms.append(((value.float()-target).square()/energy).mean())
    if not terms:
        raise ValueError('No compacted features were captured.')
    return torch.stack(terms).mean()
