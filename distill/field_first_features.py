"""Audit all five first-phase channels, separately from the height-only guard.

The first four latent channels carry decoder features, not physical heights.
Their positioned-band errors and amplitudes must not be reported in metres or
confused with a hydrology simulation. This diagnostic never forces the model.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from distill.common import DATA,atomic_json,external_path
from distill.field_base import FieldBase
from distill.field_phase import FieldPhaseBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.wet_mountains import ROOT,digest
from distill.wet_mountain_progress import selected_validation


def compare_channels(candidate,reference,variance,powers):
    rows=[]
    for channel,(a,b) in enumerate(zip(candidate,reference)):
        bands=[]
        for index,sigma in enumerate((1.,2.,4.)):
            x=a-gaussian_filter(a,sigma);y=b-gaussian_filter(b,sigma)
            ex,ey=np.mean(x*x),np.mean(y*y)
            bands.append(dict(sigma_pixels=sigma,
                positioned_mse_over_train_band_power=float(np.mean((x-y)**2)/powers[index,channel]),
                rms_ratio=float(np.sqrt(ex/ey)) if ey>0 else None,
                correlation=float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if ex>0 and ey>0 else None))
        rows.append(dict(channel=channel,role='decoder-feature' if channel<4 else 'signed-root-height',
            mse_over_train_variance=float(np.mean((a-b)**2)/variance[channel]),bands=bands))
    return rows


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    statistics=DATA/'eval/native-decoded-probe/channel-statistics.json'
    band_statistics=DATA/'eval/native-phase-train-band-statistics.json'
    variance=np.asarray(json.loads(statistics.read_text())['variance'])
    bands=json.loads(band_statistics.read_text())
    if bands['split']!='train':raise ValueError('TRAIN-only calibration required.')
    powers=np.asarray(bands['phases']['first']['band_power'])
    result=dict(note=__doc__,source_sha256=digest(__file__),rows=[],accepted=False,
        statistics_sha256=digest(statistics),band_statistics_sha256=digest(band_statistics),
        gpu=torch.cuda.get_device_name())
    for checkpoint in args.checkpoints:
        sha=digest(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cls={'native-field':FieldBase,'native-field-phase':FieldPhaseBase}[saved['architecture']]
        model=cls(saved['config']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['ema']);step=saved['step'];architecture=saved['architecture'];del saved
        for coarse in ('original','coarse4','coarse2'):
            for record in selected_validation(ROOT,coarse)[0]:
                scene=FieldScene(ROOT/record['file'],'cuda')
                predicted=field_scene_forward(model,scene)['initial192'][0].cpu().numpy().astype(np.float64)
                actual=scene.initial.cpu().numpy().astype(np.float64)
                channels=compare_channels(predicted,actual,variance,powers)
                result['rows'].append(dict(checkpoint=str(checkpoint),sha256=sha,architecture=architecture,
                    step=step,seed=scene.metadata['seed'],coarse=coarse,channels=channels,
                    feature_mse=float(np.mean([r['mse_over_train_variance'] for r in channels[:4]])),
                    all_channel_mse=float(np.mean([r['mse_over_train_variance'] for r in channels]))))
        del model,scene;gc.collect();torch.cuda.empty_cache()
    atomic_json(output,result)
    output.with_suffix('.txt').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),rows=len(result['rows']))),flush=True)


if __name__=='__main__':main()
