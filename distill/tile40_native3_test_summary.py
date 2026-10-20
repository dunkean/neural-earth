"""TEST-only metrics,copied without numeric changes from the frozen VAL formulas.

Explicit TEST schema;no case relabelling and no training use.
"""
import math
import statistics
from pathlib import Path
import numpy as np
from distill.tile40_quality_measure import measure
from distill.tile40_conditioning import CLIMATES,RELIEFS

def terrain_summary_test(report):
    """Retain mountain/climate failures even when the aggregate improves."""
    if len(report['rows']) != 30 or any(r['case']['split']!='test' for r in report['rows']):
        raise ValueError('Selection requires all30 TEST cases and excludes TRAIN/VAL.')
    seen = set()
    for row in report['rows']:
        case = row['case']
        key = (case['seed'],case['relief'],case['climate'])
        if key in seen:
            raise ValueError('Duplicate TEST case.')
        seen.add(key)
    if any(sum(r['case']['relief']==relief and r['case']['climate']==climate for r in report['rows'])!=2
           for relief in RELIEFS for climate in CLIMATES):
        raise ValueError('Two TEST geometries per relief/climate are required.')
    score,groups = measure(report)
    climates = {}
    for climate in CLIMATES:
        rows = [r for r in report['rows'] if r['case']['relief']=='mountains' and r['case']['climate']==climate]
        if len(rows)!=2:
            raise ValueError('Two mountain geometries per climate are required.')
        spectral, directional, slope = [],[],[]
        for row in rows:
            m = row['metrics']
            spectral.append(statistics.mean(abs(math.log(max(b['amplitude_ratio'],1e-6))) for b in m['bands']))
            c = m['directional_coherence']
            directional.append(abs(math.log(max(c['prediction'],1e-6)/max(c['reference'],1e-6))))
            slope.append(m['prediction_stats']['slope_p90_degrees']-m['reference_stats']['slope_p90_degrees'])
        responses = []
        for row in rows:
            temperate = next(r for r in report['rows'] if r['case']['seed']==row['case']['seed'] and
                r['case']['relief']=='mountains' and r['case']['climate']=='temperate')
            errors = []
            for band,base in zip(row['metrics']['bands'],temperate['metrics']['bands'],strict=True):
                predicted = math.log(max(band['prediction_rms_m'],1e-6)/max(base['prediction_rms_m'],1e-6))
                reference = math.log(max(band['reference_rms_m'],1e-6)/max(base['reference_rms_m'],1e-6))
                errors.append(abs(predicted-reference))
            responses.append(statistics.mean(errors))
        climates[climate] = dict(morphology_distance=statistics.mean(a+b for a,b in zip(spectral,directional)),
            spectral_distance=statistics.mean(spectral),directional_distance=statistics.mean(directional),
            slope_p90_delta_degrees=statistics.mean(slope),
            climate_spectral_response_error=statistics.mean(responses))
    return dict(score=score,groups=groups,mountain_climates=climates,
        scope='Spectral,directional and slope diagnostics;images and physical climate-response review remain required.')


def physical_summary_test(report,directory):
    summary=terrain_summary_test(report)
    summary['height_mae_m']=statistics.mean(row['metrics']['height_mae_m'] for row in report['rows'])
    summary['mountain_band_correlation']={climate:statistics.mean(
        row['paired_spatial_band_correlation'] for row in report['rows']
        if row['case']['relief']=='mountains' and row['case']['climate']==climate) for climate in CLIMATES}
    mountains=[row for row in report['rows'] if row['case']['relief']=='mountains']
    maps={}
    for row in mountains:
        case=row['case']
        with np.load(Path(directory)/case['file'],allow_pickle=False) as values:
            maps[(case['seed'],case['climate'])]=(values['height_m'].astype(np.float64),values['reference_height_m'].astype(np.float64))
    responses={}
    for climate in CLIMATES:
        if climate=='temperate':continue
        rows=[]
        for seed in sorted({row['case']['seed'] for row in mountains}):
            actual,reference=maps[(seed,climate)]
            temperate,target=maps[(seed,'temperate')]
            a,b=actual-temperate,reference-target
            denominator=max(float(np.mean(b*b)),1.)
            ac,bc=a-a.mean(),b-b.mean()
            norm=float(np.linalg.norm(ac)*np.linalg.norm(bc))
            rows.append(dict(relative_rmse=float(np.sqrt(np.mean((a-b)**2)/denominator)),
                correlation=float(np.sum(ac*bc)/norm) if norm>1e-8 else 0.))
        responses[climate]={key:statistics.mean(row[key] for row in rows) for key in rows[0]}
    summary['climate_height_response']=responses
    summary['quality_gap']=(summary['groups']['mountains']+
        .25*(1-min(summary['mountain_band_correlation'].values()))+
        .25*statistics.mean(row['relative_rmse'] for row in responses.values()))
    # Numeric proximity to the original,not automatic visual acceptance.
    summary['near_original_numeric']=(summary['groups']['mountains']<.05 and
        all(value>=.95 for value in summary['mountain_band_correlation'].values()) and
        all(value['climate_spectral_response_error']<=.025 for value in summary['mountain_climates'].values()) and
        all(value['relative_rmse']<=.35 and value['correlation']>=.9 for value in responses.values()))
    return summary
