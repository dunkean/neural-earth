"""Per-world native-base telemetry without altering its baked convolution path."""
import torch


class NativeLiveBase(torch.nn.Module):
    def __init__(self,model,counts):
        super().__init__()
        self.model=model
        self.counts=counts
        self.eval()

    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(self.model,name)

    def note(self,inputs):
        self.counts['calls']+=1
        self.counts['output_pixels']+=inputs.shape[0]*inputs.shape[-2]*inputs.shape[-1]

    def forward(self,inputs,*args,**kwargs):
        self.note(inputs)
        return self.model(inputs,*args,**kwargs)

    def forward_with_embeddings(self,inputs,*args,**kwargs):
        self.note(inputs)
        return self.model.forward_with_embeddings(inputs,*args,**kwargs)


def install(world,model):
    counts=getattr(world,'_distill_counts',{})
    counts.setdefault('base',dict(calls=0,output_pixels=0))
    world._distill_counts=counts
    world.base_model=NativeLiveBase(model,counts['base'])
