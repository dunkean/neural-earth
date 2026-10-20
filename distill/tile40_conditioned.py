"""Inject deployable coarse elevation/climate context throughout the generator."""
import torch
from torch import nn
import torch.nn.functional as F
from distill.student import Student

ARCHITECTURE='coarse-film-v1'


class Modulation(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.affine=nn.Conv2d(6,2*width,1)
        nn.init.zeros_(self.affine.weight);nn.init.zeros_(self.affine.bias)

    def forward(self,features,conditioning):
        scale,shift=self.affine(F.interpolate(conditioning,size=features.shape[-2:],mode='nearest')).chunk(2,dim=1)
        return features*(1+torch.tanh(scale))+shift


class ConditionedStudent(Student):
    conditioning_architecture=ARCHITECTURE

    def __init__(self,config):
        super().__init__(config)
        if config.stage!='base' or config.solver_steps:
            raise ValueError('Explicit compact base generator required.')
        widths=[config.width*min(2**i,2) for i in range(config.depth)]
        self.enc_mod=nn.ModuleList(Modulation(w) for w in widths)
        self.mid_mod=nn.ModuleList(Modulation(widths[-1]) for _ in config.dilations)
        self.dec_mod=nn.ModuleList(Modulation(w) for w in widths[-2::-1])

    def forward(self,inputs):
        if inputs.shape[-1]%self.alignment or inputs.shape[-2]%self.alignment:
            raise ValueError('Input alignment changed.')
        conditioning=inputs[:,10:16]
        x=self.stem(inputs);skips=[]
        for level,(block,mod) in enumerate(zip(self.encoders,self.enc_mod)):
            x=mod(block(x),conditioning)
            if level<len(self.down):skips.append(x);x=self.down[level](x)
        for block,mod in zip(self.middle,self.mid_mod):x=mod(block(x),conditioning)
        for skip,block,mod in zip(reversed(skips),self.decoders,self.dec_mod):
            x=F.interpolate(x,scale_factor=2,mode='nearest')
            x=mod(block(torch.cat((x,skip),dim=1)),conditioning)
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.head(x.float())+self.direct(inputs.float())


def transfer(source,target):
    if source.config!=target.config:
        raise ValueError('Conditioning transfer must retain the full backbone geometry.')
    result=target.load_state_dict(source.state_dict(),strict=False)
    if result.unexpected_keys or any(not name.startswith(('enc_mod.','mid_mod.','dec_mod.')) for name in result.missing_keys):
        raise ValueError('Unexpected missing backbone weights.')
