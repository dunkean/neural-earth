"""Position-sensitive terrain proxies supplement slope and power distributions.

Topographic position is a morphology proxy, not evidence of hydrological flow.
Same-site FP32/BF16 comparisons supply its natural floor before any decision.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def correlation(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    a,b=a-a.mean(),b-b.mean()
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.clip(np.dot(a,b)/denom,-1.,1.))


def compare_shape(candidate,reference):
    candidate,reference=np.asarray(candidate,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2 or min(candidate.shape)<32:
        raise ValueError('Matching two-dimensional terrain fields of at least32 pixels required.')
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError('Finite terrain fields required.')
    land=reference>0
    rows=[]
    for sigma in (2.,4.,8.,16.):
        margin=int(np.ceil(2*sigma))
        if min(reference.shape)<=2*margin:
            continue
        valid=land.copy()
        valid[:margin]=valid[-margin:]=False
        valid[:,:margin]=valid[:,-margin:]=False
        tpi=reference-gaussian_filter(reference,sigma=sigma,mode='reflect')
        actual=candidate-gaussian_filter(candidate,sigma=sigma,mode='reflect')
        if valid.sum()<16:
            rows.append(dict(sigma_pixels=sigma,land_pixels=int(valid.sum()),available=False))
            continue
        scale=float(np.sqrt(np.mean(tpi[valid]**2)))
        actual_scale=float(np.sqrt(np.mean(actual[valid]**2)))
        valleys=valid & (tpi < -.75*scale)
        predicted=valid & (actual < -.75*scale)
        union=int((valleys|predicted).sum())
        rows.append(dict(sigma_pixels=sigma,land_pixels=int(valid.sum()),available=scale>1e-6,
            tpi_rms_m=scale,candidate_tpi_rms_m=actual_scale,
            tpi_rms_ratio=actual_scale/scale if scale>1e-6 else None,
            tpi_mae_m=float(np.mean(np.abs(actual[valid]-tpi[valid]))),
            tpi_correlation=correlation(actual[valid],tpi[valid]),
            valley_pixels=int(valleys.sum()),valley_iou=float((valleys&predicted).sum()/union) if union else None,
            valley_retention=float((valleys&predicted).sum()/valleys.sum()) if valleys.any() else None))
    gy,gx=np.gradient(reference)
    ay,ax=np.gradient(candidate)
    magnitude=np.hypot(gx,gy)
    cutoff=max(float(np.quantile(magnitude[land],.25)),1e-6) if land.any() else float('inf')
    valid=land & (magnitude>cutoff)
    actual_magnitude=np.hypot(ax,ay)
    valid &= actual_magnitude>1e-6
    direction=(gx[valid]*ax[valid]+gy[valid]*ay[valid])/(magnitude[valid]*actual_magnitude[valid])
    return dict(topographic_position=rows,gradient_direction_cosine=float(direction.mean()) if len(direction) else None,
        gradient_pixels=int(valid.sum()),note='Position-sensitive morphology proxies; calibrate against same-site FP32/BF16 floor.')
