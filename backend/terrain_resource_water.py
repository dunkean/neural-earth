"""Fused resource projection of the shared monthly water-balance equations.

The forcing and parameters are owned by terrain_hydrology_balance. This kernel
keeps each cell's periodic solve local instead of allocating a dozen world-sized
temporaries per iteration. The Python fallback solves the same analytic map;
the independent shared NumPy solver remains the tolerance reference.
"""
import math
import numpy as np
try:
    from numba import njit,prange,set_num_threads
except ImportError:
    njit=None
    prange=range
    def set_num_threads(count):pass


def _solve(forcing,cap,capacity,recharge,groundwater_days,tolerance,max_years,days):
    n=forcing.shape[-1];output=np.empty((4,12,n),np.float32)
    decay=-np.expm1(-days/groundwater_days);annual_decay=math.exp(-365/groundwater_days)
    for c in prange(n):
        # The annual snow map is clamp(s + shift, lower, upper).
        lower=0.;upper=cap;shift=0.
        for month in range(12):
            net=forcing[1,month,c]-forcing[3,month,c]
            shift+=net
            lower=min(max(lower+net,0.),cap);upper=min(max(upper+net,0.),cap)
        snow=upper if shift>0 else lower
        liquid=np.empty(12,np.float64);a=np.empty(12,np.float64);b=np.empty(12,np.float64)
        snow_end=np.empty(12,np.float64)
        product=1.;addition=0.;ceiling=capacity
        for month in range(12):
            snow+=forcing[1,month,c]
            melt=min(snow,forcing[3,month,c]);snow-=melt
            overflow=max(snow-cap,0.);snow-=overflow
            snow_end[month]=snow
            liquid[month]=forcing[2,month,c]+melt+overflow
            deficit=forcing[4,month,c]-liquid[month]
            a[month]=math.exp(max(deficit,0.)/-capacity);b[month]=-min(deficit,0.)
            product=product*a[month];addition=a[month]*addition+b[month]
            ceiling=min(a[month]*ceiling+b[month],capacity)
        # A year's capped affine soil map is min(A*s+B, C).
        soil=min(addition/(1-product),ceiling) if product<1 else (ceiling if addition>0 else 0.)
        initial_soil=soil;ground=0.
        for month in range(12):
            new_soil=a[month]*soil+b[month]
            surplus=max(new_soil-capacity,0.);soil=new_soil-surplus
            ground=(ground+surplus*recharge)*(1-decay[month])
        ground=ground/(1-annual_decay);soil=initial_soil
        for month in range(12):
            new_soil=a[month]*soil+b[month]
            aet=liquid[month]+soil-new_soil
            surplus=max(new_soil-capacity,0.);soil=new_soil-surplus
            replenishment=surplus*recharge;ground+=replenishment
            baseflow=ground*decay[month];ground-=baseflow
            output[0,month,c]=forcing[4,month,c];output[1,month,c]=aet
            # Preserve the resource API's historical runoff + baseflow sum
            # (the shared runoff field already includes one baseflow term).
            output[2,month,c]=surplus-replenishment+baseflow+baseflow
            output[3,month,c]=snow_end[month]
    return output


kernel=njit(cache=True,nogil=True,parallel=True)(_solve) if njit is not None else None
ready=False


def prepare():
    global ready
    if kernel is not None:
        set_num_threads(4)
        from terrain_hydrology_balance import MONTH_DAYS
        kernel(np.zeros((5,12,2),np.float64),2000.,150.,.35,90.,.01,40,MONTH_DAYS)
        ready=True


def evaluate(temperature,precipitation,latitude):
    from terrain_hydrology_balance import BalanceParams,_seasonal_inputs,_forcing,MONTH_DAYS
    p=BalanceParams()
    _,inputs,_=_seasonal_inputs(None,None,latitude,None,temperature[6],temperature[0],
                              precipitation[6]*6,precipitation[0]*6,None,None)
    set_num_threads(4)
    forcing=_forcing(inputs,p)
    return (kernel if ready else _solve)(forcing,p.snow_cap_mm,p.soil_capacity_mm,p.recharge_fraction,p.groundwater_days,
                  p.tolerance_mm,p.max_years,MONTH_DAYS)
