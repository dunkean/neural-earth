"""Prepare two CC0 exemplars for an independent route-only material study.

Scalar luminance ranks are Gaussianized; per-mip empirical inverse-CDF rows
preserve the relative luminance distribution. This is not a full RGB/PCA or
optimal-transport implementation of Heitz/Neyret. Normal slopes and roughness
are cross-faded, with explicit limitations recorded in the receipt.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter

ECOLOGY = '--ecology' in sys.argv
BASE = Path(__file__).resolve().parents[2]/('output/render-v2-floor/by-example-ecology-causal' if ECOLOGY else 'output/render-v2-floor/by-example-causal')
SIZE = 256
NAMES = ['sandy_gravel_02', 'rock_face_03']+(['withered_grass', 'forrest_ground_01'] if ECOLOGY else [])
PERIODS = [2.56, 2.56]+([2., 2.] if ECOLOGY else [])
DETAIL = '--soil-highpass' in sys.argv or ECOLOGY
OUT = BASE/'soil-highpass' if DETAIL and not ECOLOGY else BASE


def reduce_image(a):
    return (a[::2, ::2]+a[1::2, ::2]+a[::2, 1::2]+a[1::2, 1::2])*.25


def read(name, kind):
    return np.asarray(Image.open(BASE/'assets'/f'{name}-{kind}.jpg').convert('RGB')
                      .resize((SIZE, SIZE), Image.Resampling.LANCZOS), np.float64)/255.


def prepare():
    OUT.mkdir(exist_ok=True)
    data, levels, stats = [], [], []
    table = np.ones((SIZE, SIZE, 4), np.float64)
    probability = ndtr(np.linspace(-3., 3., SIZE))
    for layer, name in enumerate(NAMES):
        color, normal, rough = read(name, 'Diffuse'), read(name, 'nor_gl'), read(name, 'Rough')[..., 0]
        gray = color@np.array([.2126, .7152, .0722])
        if DETAIL and layer != 1:
            # The terrain already supplies the broad mineral/soil mean. Keep
            # the actual exemplar's resolved grain rather than importing its
            # large coherent luminance drift into every stochastic triangle.
            gray = gray/gaussian_filter(gray, 32., mode='wrap')
        relative = gray/gray.mean()
        rank = rankdata(relative, method='average').reshape(relative.shape)
        gaussian = .5+ndtri((rank-.5)/rank.size)/6.
        n = normal*2.-1.
        # GL +V points opposite the top-to-bottom image row direction.
        slopes = np.stack((-n[..., 0]/np.maximum(.25, n[..., 2]),
                           n[..., 1]/np.maximum(.25, n[..., 2])), -1)
        slopes -= slopes.mean((0, 1))
        rough_delta = rough-rough.mean()
        surface = np.dstack((gaussian, slopes, rough_delta))
        surface_mips, gaussian_sigma = [], []
        mip, original = surface, relative
        level = 0
        while True:
            surface_mips.append(mip.copy())
            sigma = float(mip[..., 0].std())
            gaussian_sigma.append(sigma)
            table[level, :, layer] = np.quantile(original, probability)
            if len(mip) == 1:
                break
            mip, original = reduce_image(mip), reduce_image(original)
            level += 1
        data.append(surface_mips)
        stats.append(dict(asset=name, layer=layer, meanSourceRGB=color.mean((0, 1)).tolist(),
                          meanSourceGray=float(gray.mean()), relativeMean=float(relative.mean()),
                          gaussianMean=float(gaussian.mean()), gaussianSigmaByMip=gaussian_sigma,
                          normalSlopeStd=slopes.std((0, 1)).tolist(), roughMean=float(rough.mean())))
    # One array has two material layers plus the inverse-CDF lookup layer.
    # The lookup occupies rows0..8 of mip0 and is never mip-sampled itself.
    for level in range(len(data[0])):
        size = SIZE >> level
        lut = table if level == 0 else np.ones((size, size, 4), np.float64)
        array = np.stack([mips[level] for mips in data]+[lut]).astype('<f2')
        file = OUT/f'material-array-mip-{level}.bin'
        file.write_bytes(array.tobytes())
        levels.append(dict(level=level, size=size, file=file.name, bytes=file.stat().st_size,
                           sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    meta = dict(size=SIZE, layers=len(NAMES)+1, materialCount=len(NAMES), inverseLayer=len(NAMES), format='rgba16float', mipLevels=len(levels), levels=levels,
                materialPeriod_m=2.56, materialPeriods_m=PERIODS, materials=stats, allocationBytes=sum(l['bytes'] for l in levels),
                channels='material:R Gaussianized relative luminance; GB centered tangent slopes; A roughness delta. Final array layer: per-material RGBA inverse-CDF rows by mip.',
                highpassSigma_m=[period/8 if DETAIL and i != 1 else None for i, period in enumerate(PERIODS)],
                method='Independent scalar rank Gaussianization and3-patch triangle variance-preserving blend; empirical per-mip inverse CDF. Local macro RGB stays authoritative.',
                limitations=['No fullRGB PCA or3D optimal transport; color hue intentionally discarded to preserve terrain palette',
                             'Normal slopes/roughness use variance-adjusted linear cross-fades, not their own histogram transforms',
                             'Per-mip empirical inverseCDF is a prototype alternative to the2019 convolution LUT prefilter; trilinear cross-level approximation needs validation',
                             '2.56m shader period approximates the2.5m and2.7m official exemplar widths'])
    (OUT/'material-array.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(size=SIZE, allocationBytes=meta['allocationBytes'], materials=stats), indent=2))


if __name__ == '__main__':
    prepare()
