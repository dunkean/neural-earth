"""Two teacher field targets, positional bands, central-window KD and crop seams."""
import torch

from distill.native_basis_training import capture_blocks
from distill.native_feature_kd import representation_loss
from distill.native_phase_loss import band_terms


@torch.no_grad()
def teacher_features(teacher,scene):
    with capture_blocks(teacher) as features:
        teacher(scene.teacher_windows['inputs'].bfloat16(),noise_labels=scene.teacher_windows['labels'].bfloat16(),
                conditional_inputs=[scene.teacher_windows['condition'].bfloat16()])
    return {k:v.detach() for k,v in features.items()}


def central_kd(result,reference):
    actual,target={},{}
    for phase in (0,1):
        fields=result['features'+str(phase)]
        if fields.keys()!=reference.keys():raise ValueError('Matching37 teacher field block coordinates required.')
        for name,value in fields.items():
            spacing=256//value.shape[-1]
            width=32//spacing;start=112//spacing
            begin=16//spacing
            actual[f'{phase}:{name}']=value[:,:,start:start+width,start:start+width]
            target[f'{phase}:{name}']=reference[name][phase:phase+1,:,begin:begin+width,begin:begin+width]
    return representation_loss(actual,target)


def field_loss(result,scene,variance,calibration,mountain=False):
    scale=variance[None,:,None,None]
    first=result['initial192'];last=result['core']
    initial_target=scene.initial[None].detach();final_target=scene.target[None].detach()
    differences=[(a-b)/scale.sqrt() for a,b in ((first,initial_target),(last,final_target))]
    terms=dict(initial=torch.mean(differences[0].square()),final=torch.mean(differences[1].square()),
        gradient=torch.stack([d.diff(dim=a).square().mean() for d in differences for a in (-1,-2)]).mean())
    for name,pred,target,channels,power in (
        ('first',first,initial_target,slice(0,4),calibration['first_power'][:,:4]),
        ('second',last,final_target,slice(4,5),calibration['second_power'][:,4:])):
        position,energy=band_terms(pred[:,channels],target[:,channels],variance[channels],power)
        terms[name+'_position']=position;terms[name+'_energy']=energy
    energy_weight=.1 if mountain else .05
    loss=.5*(terms['initial']+terms['final'])+.05*terms['gradient']+\
        .1*(terms['first_position']+terms['second_position'])+\
        energy_weight*(terms['first_energy']+terms['second_energy'])
    return loss,terms


def shared_crop_loss(first,second,axis,variance):
    a,b=(first[:,:,:,32:],second[:,:,:,:96]) if axis=='x' else (first[:,:,32:,:],second[:,:,:96,:])
    return ((a-b).square()/variance[None,:,None,None]).mean()
