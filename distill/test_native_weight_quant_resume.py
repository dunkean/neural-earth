"""Resume must use raw weights and preserve the next optimizer update."""
import copy
import unittest

import torch

from distill.native_weight_quant_resume import restore_training
from distill import test_native_weight_quant as fixtures


class NativeWeightQuantResumeTests(unittest.TestCase):
    def test_raw_ema_and_adam_continue_without_reinitialization(self):
        torch.set_num_threads(2);torch.manual_seed(72)
        _,model=fixtures.NativeWeightQuantTests().model();model.requires_grad_(True)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-6,betas=(.9,.99))
        # Populate nonzero moments and a real step counter before serialization.
        def update(m,opt):
            opt.zero_grad(set_to_none=True)
            sum(p.square().mean() for p in m.parameters()).backward();opt.step()
        update(model,optimizer)
        ema=copy.deepcopy(model)
        with torch.no_grad():
            next(ema.parameters()).add_(.01)
        source=dict(config=model.config,architecture='native-weight-quant',scheme=model.scheme,
                    block=model.block,quantized_layers=model.quantized_layers,
                    model=copy.deepcopy(model.state_dict()),ema=copy.deepcopy(ema.state_dict()),
                    optimizer=copy.deepcopy(optimizer.state_dict()),rng=torch.get_rng_state(),
                    cuda_rng=[],step=6000,feature_weight=.2,contract={})
        raw_restored,ema_restored,opt_restored=restore_training(source,'cpu')
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,raw_restored.state_dict()[name]))
            self.assertTrue(torch.equal(ema.state_dict()[name],ema_restored.state_dict()[name]))
        self.assertFalse(torch.equal(next(raw_restored.parameters()),next(ema_restored.parameters())))
        update(model,optimizer);update(raw_restored,opt_restored)
        for a,b in zip(model.parameters(),raw_restored.parameters()):self.assertTrue(torch.equal(a,b))
        for key,slots in optimizer.state_dict()['state'].items():
            for name,value in slots.items():self.assertTrue(torch.equal(value,opt_restored.state_dict()['state'][key][name]))
        self.assertEqual(opt_restored.param_groups[0]['lr'],3e-6)
        for name in ema_restored.quantized_layers:
            self.assertIs(ema_restored.net.get_submodule(name)._weight_quant_owner,ema_restored)
        incomplete={k:v for k,v in source.items() if k!='model'}
        with self.assertRaises(ValueError):restore_training(incomplete,'cpu')


if __name__=='__main__':unittest.main()
