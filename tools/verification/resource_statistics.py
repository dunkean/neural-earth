"""Statistical contract for resource compiler revisions (no timing or GPU IO)."""
import numpy as np


def world_statistics(fields,deposits):
    import terrain_resources as R
    area=np.asarray(fields['area_m2'])*fields['land']
    weight=area/area.sum()
    metals=R.METALS
    result=dict(resources={},families={},rocks={n:float(np.sum(fields['rock_'+n]*weight)) for n in R.ROCKS},
                means={n:float(np.sum(fields[n]*weight)) for n in ('npp_g_m2','cereal_suitability','cereal_net_t_year')})
    if hasattr(deposits,'arrays'):
        a=deposits.arrays
        for k,name in enumerate(deposits.families):
            result['families'][name]=int(np.count_nonzero(a['family']==k))
        for m in metals:
            k=deposits.metals.index(m)
            contained,recoverable=a['contained'][:,k],a['recoverable'][:,k]
            result['resources'][m]=dict(count=int(np.count_nonzero(contained)),recoverable_count=int(np.count_nonzero(recoverable)),
                contained=int(contained.sum()),recoverable=int(recoverable.sum()))
    else:
        for d in deposits:
            family=d['family'];result['families'][family]=result['families'].get(family,0)+1
        for m in metals:
            c=np.array([d['contained'].get(m,0) for d in deposits],np.int64)
            r=np.array([d['medieval_recoverable'].get(m,0) for d in deposits],np.int64)
            result['resources'][m]=dict(count=int(np.count_nonzero(c)),recoverable_count=int(np.count_nonzero(r)),
                                      contained=int(c.sum()),recoverable=int(r.sum()))
    return result


def compare_statistics(reference,current):
    """Return signed changes; raises if a stated acceptance bound is exceeded."""
    deltas=dict(resources={},families={},rocks={},means={})
    for group in ('resources','families','means'):
        tolerance=.02 if group=='means' else .10
        for name,old in reference[group].items():
            before=old if isinstance(old,dict) else {'value':old}
            now=current[group][name] if isinstance(old,dict) else {'value':current[group][name]}
            change={}
            for key,value in before.items():
                relative=(now[key]-value)/value if value else float(now[key] != 0)
                if abs(relative)>tolerance:
                    raise AssertionError(f'{group}/{name}/{key}: {relative:+.3%} exceeds {tolerance:.0%}')
                change[key]=relative
            deltas[group][name]=change if isinstance(old,dict) else change['value']
    for name,old in reference['rocks'].items():
        change=current['rocks'][name]-old
        if abs(change)>.02: raise AssertionError(f'rock/{name}: {change:+.6f} exceeds 0.02')
        deltas['rocks'][name]=change
    return deltas
