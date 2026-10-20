import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from distill.tile40_native3 import Native3Base, codes_for, export_packed, fake_gaussian, initial_scale, read_packed, represented_scale
from distill.tile40_native3_prepare import selected_cases


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = dict(test=True)
        self.enc = nn.ModuleDict({'first':nn.Conv2d(5,8,3)})
        self.conv = nn.Conv2d(8,8,3)
        self.projection = nn.Linear(8,8,bias=False)
        self.noise_linear = nn.Linear(8,8,bias=False)
        self.conditional_layers = nn.ModuleList([nn.Linear(8,8,bias=False)])
        self.out_conv = nn.Conv2d(8,5,1)


class Native3Tests(unittest.TestCase):
    def test_mixed_packing_roundtrip_and_hash(self):
        torch.manual_seed(13)
        model = Native3Base(TinyNet(),block=32)
        self.assertEqual(model.quant_specs['conv.weight']['bits'],3)
        self.assertEqual(model.quant_specs['projection.weight']['bits'],4)
        self.assertNotIn('conditional_layers.0.weight',model.quant_specs)
        self.assertNotIn('noise_linear.weight',model.quant_specs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'base'
            manifest = export_packed(model,root)
            state,_ = read_packed(root)
            for name,value in model.net.state_dict().items():
                if name in model.quant_specs:
                    spec = model.quant_specs[name]
                    scale = getattr(model,spec['scale_name']).clamp_min(1e-7).half()
                    expected = codes_for(value,scale,spec['bits'],32)[1].bfloat16()
                else:
                    expected = value.bfloat16()
                self.assertTrue(torch.equal(state[name],expected),name)
            self.assertLess(manifest['total_artifact_bytes'],100000000)
            data = bytearray((root/'weights.bin').read_bytes());data[0]^=1
            (root/'weights.bin').write_bytes(data)
            with self.assertRaises(ValueError):read_packed(root)

    def test_weight_and_scale_gradients(self):
        weight = torch.randn(17,19,requires_grad=True)
        scale = initial_scale(weight,3,32).float().requires_grad_(True)
        q = fake_gaussian(weight,scale.half(),3,32)
        self.assertTrue(torch.equal(q.detach(),codes_for(weight,scale.half(),3,32)[1].detach()))
        q.sum().backward()
        self.assertTrue(torch.equal(weight.grad,torch.ones_like(weight)))
        self.assertTrue(torch.isfinite(scale.grad).all())
        self.assertGreater(float(scale.grad.abs().sum()),0)

    def test_strict_size_limit(self):
        model = Native3Base(TinyNet(),block=32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'oversize'
            with self.assertRaises(ValueError):export_packed(model,path,limit=10)
            self.assertFalse(path.exists())

    def test_actual_metadata_size_is_enforced_even_without_header_reserve(self):
        model=Native3Base(TinyNet(),block=32)
        with tempfile.TemporaryDirectory() as directory:
            normal=Path(directory)/'normal'
            manifest=export_packed(model,normal)
            with self.assertRaises(ValueError):
                export_packed(model,Path(directory)/'too-small',limit=manifest['bytes']+1,header_reserve=0)

    def test_scale_gradient_does_not_underflow_through_fp16(self):
        scale=torch.tensor([1.0002],requires_grad=True)
        actual=represented_scale(scale)
        self.assertTrue(torch.equal(actual.detach(),scale.detach().half().float()))
        (actual*1e-10).sum().backward()
        self.assertGreater(float(scale.grad[0]),0)

    def test_paired_loss_preserves_climate_difference(self):
        from distill.tile40_native3_bank import paired_loss
        target=torch.stack([torch.full((5,8,8),v) for v in (0.,1.,2.,4.)])
        batch=dict(target=target,phase=torch.tensor([0,0,1,1]))
        stats=dict(variance=[[1.]*5]*2,climate_power=[[1.]*5]*2)
        exact,terms=paired_loss(target,batch,stats)
        self.assertEqual(float(exact),0)
        self.assertEqual(float(terms['climate_pair']),0)
        collapsed=target.reshape(2,2,5,8,8).mean(1,keepdim=True).expand(-1,2,-1,-1,-1).reshape_as(target)
        loss,terms=paired_loss(collapsed,batch,stats)
        self.assertGreater(float(loss),0)
        self.assertGreater(float(terms['climate_pair']),0)

    def test_train_statistics_exclude_validation_and_check_shared_noise(self):
        from distill.common import atomic_json,atomic_write
        from distill.native_weight_quant import file_sha
        from distill.tile40_conditioning import CLIMATES
        from distill.tile40_native3_bank import ClimateBank
        with tempfile.TemporaryDirectory() as directory:
            def make(name,val_offset,bad_noise=False):
                root=Path(directory)/name;root.mkdir()
                cases=[dict(split=split,seed=seed,relief='mountains',climate=climate,
                    file=f'{split}-{climate}.npz') for split,seed in [('train',81),('val',91)] for climate in CLIMATES]
                atomic_json(root/'manifest.json',dict(cases=cases))
                signature=file_sha(root/'manifest.json')
                for index,case in enumerate(cases):
                    target=np.full((2,5,8,8),float(index%5)+(val_offset if case['split']=='val' else 0),np.float32)
                    inputs=np.zeros_like(target)
                    if bad_noise and index==1:inputs[0]+=1
                    meta=dict(case=case,capture_manifest_sha256=signature)
                    path=root/case['file']
                    atomic_write(path,lambda handle:np.savez(handle,inputs=inputs,target=target,
                        labels=np.array([1.5625,.609375],np.float32),condition=np.zeros((2,58),np.float32),
                        metadata=np.array(json.dumps(meta))))
                    atomic_json(path.with_suffix('.json'),dict(file_sha256=file_sha(path)))
                return ClimateBank(root)
            self.assertEqual(make('a',100).statistics(),make('b',10000).statistics())
            with self.assertRaisesRegex(ValueError,'initial noise'):
                make('c',100,bad_noise=True).statistics()

    def test_geometry_climate_grouping_and_reserved_test(self):
        from distill.tile40_conditioning import CLIMATES,RELIEFS
        cases = [dict(split=split,seed=1000+i,relief=relief,climate=climate)
                 for split,i in [('train',0),('val',1),('test',2)]
                 for relief in RELIEFS for climate in CLIMATES]
        selected = selected_cases(dict(cases=cases),1,1)
        self.assertEqual(len(selected),30)
        self.assertEqual({r['split'] for r in selected},{'train','val'})
        for index in range(0,len(selected),5):
            group = selected[index:index+5]
            self.assertEqual(len({r['seed'] for r in group}),1)
            self.assertEqual({r['climate'] for r in group},set(CLIMATES))

    def test_score_cadence_waits_for_both_versions_and_recovers_final_odd_checkpoint(self):
        from distill.tile40_native3_watch import briefing_marks
        history=[dict(step=0,selection='raw'),dict(step=1000,selection='ema'),
            dict(step=2000,selection='raw'),dict(step=1000,selection='raw')]
        self.assertEqual(briefing_marks(history),[])
        history.append(dict(step=2000,selection='ema'))
        self.assertEqual(briefing_marks(history),[2000])
        history += [dict(step=3000,selection=kind) for kind in ('raw','ema')]
        self.assertEqual(briefing_marks(history),[2000])
        self.assertEqual(briefing_marks(history,terminal=True),[2000,3000])


if __name__=='__main__':unittest.main()
