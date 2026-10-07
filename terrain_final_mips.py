"""Bounded physical mips of already committed native DEM tiles.

Parents use canonical child ownership and globally aligned sample cells. No NN
is called, and no learned coarse/latent approximation is described as this mip.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class MipPlan:
    step: int
    size: int
    x0: int
    y0: int
    children: tuple


def plan_mip(lod, tx, ty, tile=256, halo=24, max_pixels=2*1024**2):
    if not 1<=lod<=12:
        return None
    step=2**lod
    size=tile+2*halo
    if (size*step)**2>max_pixels:
        return None
    x0=(tx*tile-halo)*step
    y0=(ty*tile-halo)*step
    x1,y1=x0+size*step,y0+size*step
    children=tuple((x,y) for y in range(math.floor(y0/tile),math.ceil(y1/tile))
                   for x in range(math.floor(x0/tile),math.ceil(x1/tile)))
    return MipPlan(step,size,x0,y0,children)


def read_mip(plan, read_child, tile=256, halo=24):
    """read_child(tx,ty) returns a committed native halo array, or None.

    A missing contributor returns None. The caller publishes only a complete
    block mean, retaining signed metres and the original zero marine datum.
    """
    fine_size=plan.size*plan.step
    fine=np.empty((fine_size,fine_size),np.float32)
    for tx,ty in plan.children:
        array=read_child(tx,ty)
        if array is None:
            return None
        if array.shape!=(tile+2*halo,tile+2*halo) or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid native DEM contributor')
        x0=max(plan.x0,tx*tile)
        y0=max(plan.y0,ty*tile)
        x1=min(plan.x0+fine_size,(tx+1)*tile)
        y1=min(plan.y0+fine_size,(ty+1)*tile)
        fine[y0-plan.y0:y1-plan.y0,x0-plan.x0:x1-plan.x0]=array[
            y0-ty*tile+halo:y1-ty*tile+halo,x0-tx*tile+halo:x1-tx*tile+halo]
    return np.ascontiguousarray(fine.reshape(plan.size,plan.step,plan.size,plan.step).mean(axis=(1,3)),np.float32)
