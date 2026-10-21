import unittest
import torch
from torch import nn
import torch.nn.functional as F
from distill.tile40_original_decoder_control import OriginalDecoderControl


class Teacher(nn.Module):
    def forward(self,latents,gaussian):
        assert latents.shape==(1,5,64,64)
        assert torch.count_nonzero(latents[:,4:])==0
        return F.interpolate(latents[:,:1],size=(512,512),mode='nearest')+gaussian


class ControlTests(unittest.TestCase):
    def test_original_control_recovers_generated_fields_and_retains_gradient(self):
        model=OriginalDecoderControl('cpu',Teacher())
        fields=torch.randn(1,4,64,64,requires_grad=True)
        noise=torch.randn(1,1,512,512)
        inputs=torch.cat((noise,F.interpolate(fields,size=(512,512),mode='nearest'),torch.zeros(1,11,512,512)),dim=1)
        actual=model(inputs)
        torch.testing.assert_close(actual,F.interpolate(fields[:,:1],size=(512,512),mode='nearest')+noise)
        actual.mean().backward();self.assertTrue(torch.isfinite(fields.grad).all())


if __name__=='__main__':unittest.main()
