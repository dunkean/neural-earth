"""Lattice diagnostics, deliberately separate from the visual acceptance decision."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def detect(height, resolution, row0, col0, gradient=None, floor=-6):
    h = np.asarray(height, np.float64)
    gy, gx = np.gradient(h, resolution) if gradient is None else (gradient[...,1],gradient[...,0])
    normal = np.stack((-gx, -gy, np.ones_like(h)))
    normal /= np.sqrt(np.sum(normal*normal, axis=0))
    results = {}
    for name, period in {'tile': 256., 'parent': 2., 'native30m': 30/resolution,
                         'parent2': 4., 'parent3': 8., 'parent4': 16.,'floor':max(1.,30*2**floor/resolution)}.items():
        axes = []
        for axis, origin in ((1, col0), (0, row0)):
            field = np.moveaxis(normal, axis+1, -1)
            n = field.shape[-1]
            jumps = np.sqrt(np.sum(np.diff(field, axis=-1)**2, axis=0))
            indexes = np.arange(n-1)+origin+1
            phase = np.remainder(indexes-(0 if name == 'tile' else period/2), period)
            edge = np.minimum(phase, period-phase) < .1
            interior = ~edge
            discontinuity = float(jumps[..., edge].mean()/max(jumps[..., interior].mean(), 1e-12)) if edge.any() and interior.any() else None
            # A DC/linear trend must not be mistaken for a one-cycle tile grid.
            power_ratios = []
            if n/period >= 3:
                x = np.linspace(-1, 1, n)
                centred = field-field.mean(axis=-1, keepdims=True)
                centred -= np.sum(centred*x, axis=-1, keepdims=True)/np.sum(x*x)*x
                psd = np.mean(abs(np.fft.rfft(centred*np.hanning(n), axis=-1))**2, axis=tuple(range(field.ndim-1)))
                for harmonic in range(1, min(4, int(period/2))+1):
                    k = round(n*harmonic/period)
                    if k < 4 or k >= len(psd):
                        continue
                    peak = np.mean(psd[max(1,k-1):min(len(psd),k+2)])
                    neighbours = [psd[j] for d in (-7,-6,-5,-4,4,5,6,7) if 1 <= (j:=k+d) < len(psd)]
                    if neighbours:
                        power_ratios.append(dict(ratio=float(peak/max(np.mean(neighbours), 1e-25)),normalAmplitude=float(2*np.sqrt(peak)/np.hanning(n).sum()),harmonic=harmonic))
            best=max(power_ratios,key=lambda v:v['ratio']) if power_ratios else {}
            axes.append(dict(powerRatio=best.get('ratio'),normalAmplitude=best.get('normalAmplitude'),harmonic=best.get('harmonic'),gradientJumpRatio=discontinuity))
        results[name] = axes
    return dict(lattices=results, normalRMS=float(np.std(normal[:2])),
                flagged=any((a['powerRatio'] or 0)>8 and (a['gradientJumpRatio'] or 0)>1.8 for axes in results.values() for a in axes))


if __name__ == '__main__':
    manifest = Path(sys.argv[1])
    records = json.loads(manifest.read_text())
    for r in records:
        h = np.fromfile(r['height'], '<f4', count=r['width']**2).reshape(r['width'], r['width'])
        gradient=np.fromfile(r['height'],'<f4',offset=r['gradientOffset']).reshape(r['width'],r['width'],2) if r.get('gradientOffset') else None
        r['detector'] = detect(h, r['resolution'], r['ty']*256-r['halo'], r['tx']*256-r['halo'],gradient,r.get('floor',-6))
        gy, gx = np.gradient(h.astype(np.float64), r['resolution']) if gradient is None else (gradient[...,1],gradient[...,0])
        light = np.clip((.6*gx-.4*gy+1)/np.sqrt(gx*gx+gy*gy+1), 0, 1)
        Image.fromarray(np.uint8(light*255)).save(str(Path(r['height']).with_suffix('.normal.png')))
    manifest.with_name('grid-detector.json').write_text(json.dumps(dict(thresholds=dict(powerRatio=8,gradientJumpRatio=1.8,minResolvedCycles=3),records=records),indent=2))
    for r in records:
        print(r['site'],r['mode'],r['lod'],r['detector']['flagged'],json.dumps(r['detector']['lattices']))
