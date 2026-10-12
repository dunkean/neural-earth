"""Numerically verified cached evaluation copies; leaves the training code intact."""
import json

import numpy as np
import torch

from distill.common import DATA, REPO, append_json
from distill.native_lowrank import NativeLowrankBase
from distill.native_lowrank_inference import freeze
from distill.native_decoded import DecodedRegion
from distill.native_feature_train import balanced_windows
from distill.wet_mountains import digest


class CachedEvaluationBase(NativeLowrankBase):
    def load_state_dict(self,state,**options):
        result=super().load_state_dict(state,**options)
        cached=freeze(self)
        root=DATA/'crops/native-mixed-scenes-v2'
        rows=[r for r in json.loads((root/'manifest.json').read_text())['scenes']
              if r['metadata']['split']=='val'][:3]
        errors=[]
        with torch.no_grad():
            for index,row in enumerate(rows):
                batch=balanced_windows(DecodedRegion(root/row['file'],'cuda'),np.random.default_rng(39120+index))
                before=super().forward(batch['inputs'],batch['labels'],[batch['condition']])
                after=cached(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
                             conditional_inputs=[batch['condition'].bfloat16()]).float()
                errors.append(dict(coarse=row['coarse_variant'],max_abs=float((before-after).abs().max())))
        if any(r['max_abs']!=0. for r in errors):
            raise ValueError(f'Cached checkpoint evaluation must exactly preserve both diffusion phases: {errors}')
        self.cached_net=cached
        append_json(DATA/'eval/native-basis-cached-equivalence.jsonl',dict(ranks=self.ranks,errors=errors,
            evaluator_sha256=digest(REPO/'distill/native_basis_cached_evaluate.py'),
            inference_sha256=digest(REPO/'distill/native_lowrank_inference.py'),gpu=torch.cuda.get_device_name()))
        return result

    def forward(self,inputs,noise_labels,conditional_inputs):
        return self.cached_net(inputs.bfloat16(),noise_labels=noise_labels.bfloat16(),
            conditional_inputs=[v.bfloat16() for v in conditional_inputs]).float()


def main():
    from distill import native_basis_evaluate
    native_basis_evaluate.NativeLowrankBase=CachedEvaluationBase
    native_basis_evaluate.main()


if __name__=='__main__':main()
