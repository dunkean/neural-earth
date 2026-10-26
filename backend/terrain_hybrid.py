"""Drainage-aligned phasor heights with independently calibrated materials.

Heights always come from terrain_phasor's explicit drainage path; no physical
erosion delta enters this mode. Materials come from amplification.sample_surface
on these same heights. A lazy, request-local height provider lets the surface
reader choose its own halo first, then crops the requested heights from that
single evaluation. It is independent of tile arrival and persists no data.
"""
from __future__ import annotations

import math
import torch

import terrain_amplification as amplification
import terrain_phasor as phasor

SURFACE_CHANNELS = phasor.SURFACE_CHANNELS
SURFACE_VERSION = getattr(amplification, 'SURFACE_VERSION', 'phasor-surface-fallback')
VERSION = (f'hybrid-v1+{phasor.PHASOR_DRAINAGE_VERSION}+{SURFACE_VERSION}'
           f'+drainage-{amplification.VERSION}+canonical-native-v1')


def _contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


class _HeightProvider:
    """One enlarged read in the usual case; bounded chunks for very large halos."""

    def __init__(self, world, level, intensity, check):
        self.world, self.level, self.intensity, self.check = world, level, intensity, check
        self.reads = []

    def read(self, a1, b1, a2, b2):
        bounds = a1, b1, a2, b2
        for outer, heights, surface in self.reads:
            if _contains(outer, bounds):
                y, x = a1-outer[0], b1-outer[1]
                return (heights[y:y+a2-a1, x:x+b2-b1].contiguous(),
                        surface[:, y:y+a2-a1, x:x+b2-b1].contiguous())
        if len(self.reads) > 1:
            # sample_surface splits oversized halos into bounded rectangles.
            # Reassemble their visible intersection instead of sampling it a
            # second time when no single rectangle contains the whole view.
            ys = sorted({a1,a2} | {v for outer,_,_ in self.reads for v in (outer[0],outer[2]) if a1<v<a2})
            xs = sorted({b1,b2} | {v for outer,_,_ in self.reads for v in (outer[1],outer[3]) if b1<v<b2})
            rows, covered = [], True
            for ya,yb in zip(ys,ys[1:]):
                tiles = []
                for xa,xb in zip(xs,xs[1:]):
                    match = next(((outer,h,s) for outer,h,s in self.reads
                                  if _contains(outer,(ya,xa,yb,xb))),None)
                    if match is None:
                        covered = False
                        break
                    outer,h,s = match
                    y,x = ya-outer[0],xa-outer[1]
                    tiles.append((h[y:y+yb-ya,x:x+xb-xa],s[:,y:y+yb-ya,x:x+xb-xa]))
                if not covered:
                    break
                rows.append(tuple(torch.cat([tile[k] for tile in tiles],dim=-1) for k in (0,1)))
            if covered:
                h,s = [torch.cat([row[k] for row in rows],dim=-2).contiguous() for k in (0,1)]
                self.reads.append((bounds,h,s))
                return h,s
        if self.check is not None:
            self.check()
        if (a2-a1)*(b2-b1) <= 1024**2:
            heights, surface = phasor.sample_phasor(self.world, self.level, *bounds,
                intensity=self.intensity, drainage=True, check=self.check)
        else:
            # Surface halos may legitimately exceed the public output budget.
            # Globally evaluated pieces agree exactly at every internal edge.
            rows = []
            for y in range(a1, a2, 512):
                tiles = [self.read(y, x, min(y+512, a2), min(x+512, b2))
                         for x in range(b1, b2, 512)]
                rows.append(tuple(torch.cat([tile[k] for tile in tiles], dim=-1) for k in (0, 1)))
            heights, surface = [torch.cat([row[k] for row in rows], dim=-2) for k in (0, 1)]
        self.reads.append((bounds, heights, surface))
        return heights, surface

    def __call__(self, a1, b1, a2, b2):
        return self.read(a1, b1, a2, b2)[0]


@torch.inference_mode()
def sample_hybrid(world, level, i1, j1, i2, j2, *, intensity=1.0, check=None):
    """Return FP32 metre heights and flow/deposit/scree/rock on the fine grid.

    Intensity zero and native level zero exactly preserve sample_refined and
    return zero guides. Missing surface API falls back only to phasor guides;
    exceptions from an installed public surface reader are never hidden.
    """
    try:
        strength = float(intensity)
    except (TypeError, ValueError) as exc:
        raise ValueError('Hybrid intensity must be finite and nonnegative') from exc
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Hybrid intensity must be finite and nonnegative')
    if (not isinstance(level, int) or not 0 <= level <= 6 or
            any(not isinstance(v, int) for v in (i1, j1, i2, j2)) or
            i2 <= i1 or j2 <= j1 or (i2-i1)*(j2-j1) > 1024**2):
        raise ValueError('Invalid or oversized hybrid refinement window')
    if check is not None:
        check()
    if strength == 0 or level == 0:
        return phasor.sample_phasor(world, level, i1, j1, i2, j2,
                                   intensity=strength, drainage=True, check=check)
    # Initialize the public drainage tensor against the original source world
    # before lending the surface reader a coherent native-height view. Its
    # shared cached drainage continues to use A's unchanged native definition.
    view = phasor._drainage_world_view(world,level,i1,j1,check)
    provider = _HeightProvider(view, level, strength, check)
    reader = getattr(amplification, 'sample_surface', None)
    if reader is None:
        return provider.read(i1, j1, i2, j2)
    surface = reader(view, level, i1, j1, i2, j2, provider, check=check)
    height = provider(i1, j1, i2, j2)
    if surface.shape != (4, i2-i1, j2-j1):
        raise ValueError('Hybrid surface reader returned an invalid shape')
    if check is not None:
        check()
    return height.float().contiguous(), surface.to(device=height.device, dtype=torch.float32).clamp(0, 1).contiguous()
