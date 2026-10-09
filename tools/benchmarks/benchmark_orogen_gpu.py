"""Compare explicit Orogen GPU presets against the historical CPU defaults."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from benchmark_orogen import preview
import terrain_orogen as orogen

PRESETS = {
    'cpu': (),
    'climate': ('climate','raster'),
    'local': ('relief','post','climate','raster'),
    'full': ('relief','propagation','post','erosion','climate','raster'),
    'city': ('relief','propagation','post','climate','raster'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--detail',type=int,default=204000)
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--height',type=int,default=1024)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--preset',action='append',choices=PRESETS)
    parser.add_argument('--output',type=Path,default=Path('output/orogen/gpu-comparison'))
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(seed=str(args.seed),detail=args.detail,width=args.width,height=args.height,
                includes_nn=False,includes_persistence=False,cases=[])
    reference=None
    for preset in args.preset or PRESETS:
        options=dict(detail=args.detail,relief_pipeline='city-gpu' if preset=='city' else 'orogen',
                     **{'orogen_gpu_'+stage:True for stage in PRESETS[preset]})
        # Resolve lazy device/runtime probing before the generation timer, while
        # keeping its cost visible separately. Each case has one cold generation.
        setup=time.perf_counter();orogen.style_config('earthlike',options)
        setup=time.perf_counter()-setup
        first=orogen.generate_atlas(args.seed,width=args.width,height=args.height,options=options,include_layers=True)
        timings=[];last=first
        for repeat in range(args.repeats):
            last=orogen.generate_atlas(args.seed,width=args.width,height=args.height,options=options,include_layers=True)
            np.testing.assert_array_equal(first[0],last[0])
            for name,value in first[4].items():np.testing.assert_array_equal(value,last[4][name])
            timings.append(last[2]['generation_seconds'])
        median=statistics.median(timings)
        case=dict(preset=preset,options=options,setup_seconds=setup,cold_seconds=first[2]['generation_seconds'],
                  warm_seconds=timings,median_seconds=median,repeat_bit_exact=True,
                  diagnostics=last[2]['diagnostics'],metadata=last[2])
        if reference is None and preset=='cpu':reference=first
        if reference is not None:
            case['quality_comparison']=dict(
                coast_changed_fraction=float(np.mean((last[0]>0)!=(reference[0]>0))),
                height_mae_m=float(np.abs(last[0]-reference[0]).mean()),
                koppen_same_fraction=float(np.mean(last[4]['koppen']==reference[4]['koppen'])))
        preview(last[0]).save(args.output/(preset+'-relief.png'))
        np.savez_compressed(args.output/(preset+'-fields.npz'),height_m=last[0],
            temperature_summer=last[4]['temperature_summer'],temperature_winter=last[4]['temperature_winter'],
            precip_summer=last[4]['precip_summer'],precip_winter=last[4]['precip_winter'],koppen=last[4]['koppen'])
        report['cases'].append(case)
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(preset=preset,median_seconds=median,cold_seconds=case['cold_seconds'],
                              quality=case.get('quality_comparison'),repeat_bit_exact=True)),flush=True)


if __name__=='__main__':main()
