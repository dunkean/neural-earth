"""Render v2 authority environment, with date-independent terms evaluated once.

Same NumPy operations/order as terrain_render_v2.evaluate; CPU canonical path.
This consumes the authority footprint, never a displayed tile or forest slider.
The reference evaluator remains independent and is exercised in regression tests.
"""
import numpy as np
from terrain_render import F, smooth, gradient_noise
from terrain_render_v2 import snow_cover, perennial_snow


def months(point, gradient, tpi, ts, tw, ps, pw, seed, footprint, land_point):
    offset = np.array([seed & 255, (seed >> 8) & 255, (seed >> 16) & 255], F) * F(.37)
    province = (.72 * gradient_noise(point / F(180000) + offset, seed + 601) * (1-smooth(21600,54000,footprint))
                + .28 * gradient_noise(point / F(690000) + offset, seed + 607) * (1-smooth(82800,207000,footprint)))
    # Below LOD 5 use the full reference mosaic. At LOD 10 both amplitudes
    # are exactly zero, so allocating/hashing the unresolved noise is pointless.
    mosaic = np.zeros(point.shape[:-1], F)
    if footprint < 720:
        mosaic = (.72 * gradient_noise(land_point / F(2400), seed + 613) * (1-smooth(288,720,footprint))
                  + .28 * gradient_noise(land_point / F(670), seed + 617) * (1-smooth(80.4,201,footprint)))
    slope = np.linalg.norm(gradient, axis=-1)
    valley, ridge = smooth(0, -.10, tpi), smooth(0, .10, tpi)
    exposure = np.clip(.5+.28*gradient[..., 1]/np.maximum(.2, slope)+.22*ridge-.22*valley, 0, 1)
    threshold = np.maximum(120, 20*(ts+tw)*.5+280)
    potential = np.clip((ps+pw)/threshold*np.exp(.12*mosaic)*(1+.45*valley-.14*ridge)+0.*.8, 0, 5)
    rock = np.clip(smooth(.20,.85,slope)*(.65+.35*smooth(-.5,.9,mosaic))+.12*ridge*(1-smooth(.5,1.5,potential)),0,1)
    grow = np.maximum(ts, tw)
    wet_tropic = smooth(16,23,np.minimum(ts,tw))*smooth(1.5,3,potential)
    thermal_gate = smooth(5,9,grow+.7*mosaic+.6*valley)
    stands = np.clip(1+(.20-.15*wet_tropic)*mosaic-.17*smooth(.7,1.4,mosaic)*(1-wet_tropic),.55,1.15)
    woody = np.clip(smooth(.38,1.65,potential)*(.72+.25*wet_tropic)*thermal_gate*stands*(1-rock)+0.*.5*thermal_gate,0,.985)
    height = (2+30*smooth(.5,3,potential)*smooth(5,22,grow))*(1+.15*mosaic)*thermal_gate
    needle = np.clip(1-smooth(9,19,grow)+.25*smooth(.2,.8,province)-wet_tropic,0,1)
    snow_supply = np.clip((ps+pw)/threshold,0,5)
    snow_exposure = .65*(exposure-.5)
    perennial = perennial_snow(grow+snow_exposure,snow_supply,slope)
    out = {}
    for month in range(12):
        phase = .5-.5*np.cos(2*np.pi*((month+.5)/12))
        temperature = tw+(ts-tw)*phase+.65*(exposure-.5)+.35*mosaic
        rain = (pw+(ps-pw)*phase)*2/threshold
        moisture = np.clip((.62*potential+.38*rain)*(1-.18*smooth(16,34,temperature))+.18*valley+0.*.6,0,4)
        leaf = ((1-smooth(16,23,np.minimum(ts,tw)))*(.12+.88*smooth(-3,14,temperature))
                +smooth(16,23,np.minimum(ts,tw))*smooth(.28,.9,moisture))
        leaf = leaf+(1-leaf)*needle
        snow = np.maximum(snow_cover(tw+(ts-tw)*phase+snow_exposure,snow_supply,slope,valley),perennial)
        ice = smooth(-1.8,-8,temperature)
        grass = np.clip(smooth(.12,.8,potential)*smooth(0,9,grow)*(1-rock)*(1-snow),0,1)
        out[month] = {k:np.asarray(v,F) for k,v in dict(snow=snow,grass=grass,leaf=leaf,ice=ice,
                                                        moisture=moisture,woody=woody,height=height,needle=needle).items()}
    return out
