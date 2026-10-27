"""Area-weighted contrast of every metal, from a compiled authority NPZ.

Usage: python tools/verification/measure_resources_contrast.py CACHE.npz --out REPORT.json
Low means <1% of the fixed physical upper bound, not a fitted world quantile.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'backend'))
import terrain_resources as R


def measure(fields):
    land = np.asarray(fields['land'], bool)
    area = np.asarray(fields['area_m2'], np.float64)[land]
    result = {}
    for metal in R.METALS:
        p = R.metal_potential(fields, metal)[land]
        order = np.argsort(-p, kind='stable')
        a, v = area[order], p[order]
        cumul = np.cumsum(a) - a
        total = float(np.sum(a * v))
        upper = sum(R.metal_potential_weights(metal).values())
        row = dict(physical_upper=upper, low_threshold=.01*upper,
                   low_land_pct=float(100*area[p < .01*upper].sum()/area.sum()),
                   maximum=float(p.max()), potential_mass=total/1e6)
        for fraction in (.1, .2):
            partial = np.clip(fraction*area.sum()-cumul, 0, a)
            row[f'top{int(fraction*100)}_mass_pct'] = float(100*np.sum(partial*v)/total) if total else None
        result[metal] = row
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        result=measure(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
