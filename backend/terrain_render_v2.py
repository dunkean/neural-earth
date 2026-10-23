"""Render v2 environmental appearance, CPU overview/PNG reference.

No DEM modification. Refined climate/soil fields are appearance proxies in
world metres. Equations match web/terrain_render_v2.js; CPU integrates canopy
coverage instead of drawing individual crowns. See implementation report.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from terrain_render import F, smooth, mix, gradient_noise, parse_settings, terrain_derivatives, _hash


def clearing_target(control, humid, trop):
    """Natural-woodland appearance policy, not a measured universal gap rate."""
    return control*(.4+(.08-.4)*smooth(.8, 1.8, humid))*(1-.85*trop)


def clearing_fraction(score, threshold):
    """Bounded stand thinning; physical openings and individual gaps are separate.

    The caller handles a zero control. Decode the legacy target quantile so
    symmetric scores retain its mean before the 55% maximum reduction.
    """
    mean_share = 1/(1+np.exp(np.clip(np.asarray(threshold)*1.702/1.15, -20, 20)))
    return np.minimum(.55, 2*mean_share/(1+np.exp(np.clip(-np.asarray(score)/1.15, -20, 20))))


def stand_coverage(cover, stand, disturbed, trop):
    """Modulate effective optical density, not saturated cover directly.

    The same gap law already governs the forest-density control. This is an
    appearance population model, not a measured physical LAI. Caller cover is
    bounded by its climatic maximum below one; eligibility remains separate.
    """
    cover, stand = np.asarray(cover), np.asarray(stand)
    mosaic = .22*(1-.75*np.asarray(trop))
    density = np.exp(mosaic*stand+.05*stand*(1-cover))*(1-.3*np.asarray(disturbed)*(1-.7*np.asarray(trop)))
    return 1-np.power(np.maximum(1-cover, .001), density)


def snow_cover(temperature, potential, slope, valley=0.):
    """Continuous climate/retention fraction, without stand or bank noise."""
    return np.clip(smooth(2, -5, temperature-.8*valley)*smooth(.12, .7, potential)
                   *(1-smooth(.45, 1.25, slope)), 0, 1)


def perennial_snow(warmest, potential, slope):
    """Annual cold-climate floor with V2 physical supply/slope retention.

    The temperature interval is inherited from the perennial Render material;
    no seasonal stand noise, biome label or coastal glacier channel is used.
    """
    return np.clip(smooth(2.5, -2.5, warmest)*smooth(.12, .7, potential)
                   *(1-smooth(.45, 1.25, slope)), 0, 1)


def evaluate(point, height, gradient, tpi, ts, tw, ps, pw, sand, bank=0., date=.5, seed=0, footprint=0., land_point=None):
    """Return the same named environmental fields used by render and diagnostics."""
    land_point = point if land_point is None else land_point
    noise = lambda scale, salt: gradient_noise(land_point / F(scale), seed + salt)
    offset=np.array([seed&255,(seed>>8)&255,(seed>>16)&255],F)*F(.37)
    province = .72*gradient_noise(point/F(180000)+offset,seed+601)*(1-smooth(21600,54000,footprint))+.28*gradient_noise(point/F(690000)+offset,seed+607)*(1-smooth(82800,207000,footprint))
    mosaic = .72*noise(2400, 613)*(1-smooth(288,720,footprint))+.28*noise(670, 617)*(1-smooth(80.4,201,footprint))
    slope = np.linalg.norm(gradient, axis=-1)
    valley, ridge = smooth(0, -.10, tpi), smooth(0, .10, tpi)
    exposure = np.clip(.5+.28*gradient[..., 1]/np.maximum(.2, slope)+.22*ridge-.22*valley, 0, 1)
    phase = .5-.5*np.cos(2*np.pi*date)
    temperature = tw+(ts-tw)*phase+.65*(exposure-.5)+.35*mosaic
    threshold = np.maximum(120, 20*(ts+tw)*.5+280)
    potential = np.clip((ps+pw)/threshold*np.exp(.12*mosaic)*(1+.45*valley-.14*ridge)+bank*.8, 0, 5)
    rain = (pw+(ps-pw)*phase)*2/threshold
    moisture = np.clip((.62*potential+.38*rain)*(1-.18*smooth(16, 34, temperature))+.18*valley+bank*.6, 0, 4)
    mineral = np.clip(.5+.42*province, 0, 1)
    sand = np.clip(sand*(1-.65*smooth(.16, .7, slope))+.13*valley*(1-smooth(.5, 1.4, potential)), 0, 1)
    clay = np.clip(.18+.25*valley+.16*(1-sand)-.12*slope, 0, 1-sand)
    organic = np.minimum(1-sand-clay, np.clip(.18*smooth(.6, 2, potential)*(1-smooth(.12, .7, slope))+.13*valley, 0, .6))
    rock = np.clip(smooth(.20, .85, slope)*(.65+.35*smooth(-.5, .9, mosaic))+.12*ridge*(1-smooth(.5, 1.5, potential)), 0, 1)
    grow = np.maximum(ts, tw)
    wet_tropic = smooth(16, 23, np.minimum(ts, tw))*smooth(1.5, 3, potential)
    thermal_gate = smooth(5, 9, grow+.7*mosaic+.6*valley)
    stands = np.clip(1+(.20-.15*wet_tropic)*mosaic-.17*smooth(.7, 1.4, mosaic)*(1-wet_tropic), .55, 1.15)
    woody = np.clip(smooth(.38, 1.65, potential)*(.72+.25*wet_tropic)*thermal_gate*stands*(1-rock)+bank*.5*thermal_gate, 0, .985)
    canopy_height = (2+30*smooth(.5, 3, potential)*smooth(5, 22, grow))*(1+.15*mosaic)*thermal_gate
    needle = np.clip(1-smooth(9, 19, grow)+.25*smooth(.2, .8, province)-wet_tropic, 0, 1)
    leaf = (1-smooth(16, 23, np.minimum(ts, tw)))*(.12+.88*smooth(-3, 14, temperature))+smooth(16, 23, np.minimum(ts, tw))*smooth(.28, .9, moisture)
    leaf = leaf+(1-leaf)*needle
    snow_supply = np.clip((ps+pw)/threshold, 0, 5)
    snow_exposure = .65*(exposure-.5)
    snow = np.maximum(snow_cover(tw+(ts-tw)*phase+snow_exposure, snow_supply, slope, valley),
                      perennial_snow(grow+snow_exposure, snow_supply, slope))
    ice = smooth(-1.8, -8, temperature)
    grass = np.clip(smooth(.12, .8, potential)*smooth(0, 9, grow)*(1-rock)*(1-snow), 0, 1)
    tone = np.clip(.5+.13*mosaic-.06*valley, 0, 1)
    return dict(temperature=temperature, potential=potential, moisture=moisture, exposure=exposure,
                mineral=mineral, sand=sand, clay=clay, organic=organic, rock=rock, grass=grass,
                woody=woody, height=canopy_height, needle=needle, leaf=leaf, snow=snow, ice=ice, tone=tone, glacier=np.zeros_like(temperature))


def mineral_color(e):
    return mix(mix([.19, .19, .175], [.39, .335, .27], smooth(.08, .35, e['mineral'])),
               mix([.68, .655, .57], [.49, .37, .285], smooth(.68, .94, e['mineral'])), smooth(.34, .66, e['mineral']))


def ground_tint(base, e):
    """Bounded mineral variation of a retained ground mean, below overlays."""
    neutral = dict(e)
    neutral['mineral'] = np.full_like(e['mineral'], .5)
    ratio = np.clip(mineral_color(e)/np.maximum(mineral_color(neutral), .08), .6, 1.4)
    weight = .35*(e['rock']+(1-e['rock'])*(1-e['grass'])*(1-e['organic'])*.35)*(1-e['snow'])
    return np.asarray(base)*mix(np.ones_like(ratio), ratio, weight)


def display_shoulder(rgb):
    """Display highlight roll-off used by the hybrid GPU lighting paths."""
    rgb = np.asarray(rgb)
    peak = np.max(rgb, axis=-1)
    mapped = .8+.2*(1-np.exp(-np.maximum(peak-.8, 0)/.2))
    scale = np.where(peak > .8, mapped/np.maximum(peak, 1e-8), 1)
    return rgb*scale[..., None]


def appearance(e, footprint=0.):
    """Display contrast reveal, without changing the ecological world fields."""
    shown = dict(e)
    shown['mineral'] = .5+(e['mineral']-.5)*(1-smooth(600, 2400, footprint))
    shown['tone'] = .5+(e['tone']-.5)*(1-smooth(120, 500, footprint))
    return shown


def colors(e, diagnostic=0, footprint=0.):
    e = appearance(e, footprint)
    mineral = mineral_color(e)
    if diagnostic == 1:
        return mix(mix(mineral, [.80, .70, .49], e['sand']), [.22, .18, .13], e['organic'])
    if diagnostic == 2:
        return mix([.66, .37, .14], [.08, .40, .52], smooth(0, 2.4, e['moisture']))
    if diagnostic == 3:
        return np.stack((e['grass']*.55, e['woody']*e['leaf'], .16+e['needle']*.6), -1)
    if diagnostic == 4:
        return mix([.13, .20, .22], [.90, .96, 1], e['snow'])
    sand = mix([.73, .64, .46], mineral, np.full_like(e['sand'], .24))
    soil = mix(mineral*.8, [.24, .21, .15], e['organic']*1.2)
    soil = mix(soil, sand, e['sand'])*(1-.20*smooth(.5, 2, e['moisture']))[..., None]
    green = smooth(.25, 1.3, e['moisture'])*smooth(1, 13, e['temperature'])
    grass = mix([.53, .485, .335], [.265, .355, .175], green)
    opened = mix(mix(soil, grass, e['grass']*.88), mineral, e['rock'])
    opened = mix(opened, [.88, .91, .91], e['snow'])
    return mix(opened, canopy_color(e), e['woody']*e['leaf'])


def canopy_color(e):
    evergreen = mix([.16, .235, .14], [.12, .205, .15], e['needle'])
    senescent = mix([.34, .29, .215], [.43, .335, .205], smooth(4, 12, e['temperature']))
    canopy = mix(senescent, evergreen, smooth(.3, .9, e['leaf']))*(.85+.3*e['tone'])[..., None]
    canopy = mix(canopy, [.72, .79, .77], e['snow']*.62)
    return canopy


def _ice_cell(point, scale, seed, bisector=False):
    q = np.asarray(point, F)/F(scale)
    cell = np.floor(q).astype(np.int32)
    local = q-np.floor(q)
    first = np.full(q.shape[:-1], 10., F)
    second, identity, face = first.copy(), np.zeros_like(first), np.zeros_like(first)
    metric = np.any(bisector)
    if metric:
        first_delta, second_delta = np.zeros_like(q), np.zeros_like(q)
    for z in (-1, 0, 1):
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                offset = np.array([x, y, z], np.int32)
                hashed = _hash(cell+offset, seed)
                jitter = np.stack((hashed&1023, (hashed>>10)&1023, (hashed>>20)&1023), -1).astype(F)/F(1023)
                delta = offset.astype(F)+F(.12)+F(.76)*jitter-local
                distance = np.sum(delta*delta, -1)
                nearest = distance < first
                if metric:
                    next_second = np.where((distance < second)[..., None], delta, second_delta)
                    second_delta = np.where(nearest[..., None], first_delta, next_second)
                    first_delta = np.where(nearest[..., None], delta, first_delta)
                second = np.where(nearest, first, np.minimum(second, distance))
                first = np.minimum(first, distance)
                identity = np.where(nearest, (hashed&65535).astype(F)/F(65535), identity)
                face = np.where(nearest, np.sum(delta*np.array([.65, -.30, .70], F), -1), face)
    edge = np.sqrt(second)-np.sqrt(first)
    if metric:
        distance = (second-first)/np.maximum(2*np.sqrt(np.sum((second_delta-first_delta)**2, -1)), F(.00001))
        edge = np.where(bisector, distance, edge)
    return np.stack((edge, np.sqrt(first), identity, face), -1)


def _pack_mean(concentration):
    marginal = concentration*(.55+.50*concentration)
    return marginal+(F(.993)-marginal)*smooth(.68, .90, concentration)


def ice_material(depth, kind, e, point, footprint, seed):
    """Coverage, brightness, submerged cyan rim and cast shadow; WGSL v2Ice."""
    depth, kind = np.asarray(depth, F), np.asarray(kind, F)
    pattern = lambda scale, salt: gradient_noise(point/F(scale), seed+salt)*(1-smooth(scale*.12, scale*.3, footprint))
    freeze = np.where(kind < 1.5, smooth(-2, -9, e['temperature']),
                      np.where(kind > 2.5, smooth(-5, -13, e['temperature']), e['ice']))
    shore = smooth(900, 15, depth)
    province = pattern(7200, 631)*(1-smooth(100, 500, footprint))
    concentration = np.clip(freeze+.12*shore*freeze*(1-freeze)+.045*province*freeze*(1-freeze)*4, 0, .995)
    pack = _pack_mean(concentration)
    brightness = np.full_like(pack, .88)
    rim, shadow = np.zeros_like(pack), np.zeros_like(pack)
    if np.any(concentration > .002) and footprint < 450:
        displacement = np.stack((pattern(1800, 633), pattern(2100, 634), pattern(1600, 635)), -1)
        pack_keep = 1-smooth(140, 450, footprint)
        resolved = np.zeros_like(pack)
        local_light, local_rim = np.full_like(pack, .88), np.zeros_like(pack)
        if np.any(concentration < .90):
            warped = point+displacement*F(215)
            plate = _ice_cell(warped, 2600, seed+637)
            shard = _ice_cell(warped, 470, seed+638)
            edge = plate[..., 0]+.012*pattern(105, 639)
            aa = max(.0008, footprint/2600*.75)
            lead = .0025+.020*(1-concentration)
            large = smooth(lead-aa, lead+aa, edge)
            split = (np.mod(plate[..., 2]*F(7.31), 1) > F(.31)).astype(F)
            small_lead = .007+.09*(1-concentration)
            small_aa = max(.002, footprint/470*.8)
            shards = smooth(small_lead-small_aa, small_lead+small_aa, shard[..., 0])
            small_keep = 1-smooth(45, 135, footprint)
            subdivided = large*(1-split*(1-shards)*small_keep)
            identity = plate[..., 2]+(np.mod(shard[..., 2]*F(3.17), 1)-plate[..., 2])*split*.78
            selection = (identity >= 1-concentration).astype(F)
            resolved = subdivided*selection
            ridge = (1-smooth(lead+.006+aa, lead+.025+aa, edge))*large
            local_light = np.clip(.84+.08*plate[..., 2]+.07*ridge-.065*plate[..., 3]+.02*pattern(38, 643), .65, .99)
            local_rim = (1-smooth(lead+.004+aa, lead+.013+aa, edge))*resolved*.3
            active = concentration < .90
            resolved = np.where(active, resolved, 0)
            local_light = np.where(active, local_light, F(.88))
            local_rim = np.where(active, local_rim, 0)
        if np.any(concentration > .68):
            # Warp the common domain so both banks move without widening leads.
            jitter = np.stack((pattern(240, 639), pattern(240, 640), pattern(240, 641)), -1)*F(40)
            jitter += np.stack((pattern(70, 669), pattern(70, 670), pattern(70, 671)), -1)*F(7)
            marine = kind < 1.5
            compact = _ice_cell(point+displacement*F(65)+jitter, 1200, seed+637, bisector=marine)
            edge = compact[..., 0]
            aa = max(.001, footprint/1200*.8)
            lead = (.004+.012*(1-concentration))*(.72+.44*smooth(-.6, .6, displacement[..., 1]))
            floe = smooth(lead-aa, lead+aa, edge)
            marine_lead = (.0015+.004*(1-concentration))*(.72+.44*smooth(-.6, .6, displacement[..., 1]))
            pixel_width = max(footprint/1200, .0001)
            gap = np.clip((marine_lead+pixel_width*.5-edge)/pixel_width, 0, 1)-np.clip((-marine_lead+pixel_width*.5-edge)/pixel_width, 0, 1)
            lead = np.where(marine, marine_lead, lead)
            floe = np.where(marine, 1-gap, floe)
            opening = smooth(-.35+.50*freeze, .05+.50*freeze, displacement[..., 0]+.28*displacement[..., 2])
            open_lead = opening*np.where(kind < 1.5, F(1), F(.38))
            compact_cover = 1-(1-floe)*open_lead
            ridge = (1-smooth(lead+.009+aa, lead+.035+aa, edge))*floe
            dense = smooth(.68, .90, concentration)
            resolved = resolved+(compact_cover-resolved)*dense
            snow_grain = .012*pattern(350, 665)+.006*pattern(110, 667)
            young_ice = (1-floe)*(1-open_lead)
            compact_light = np.clip(.88+.008*(compact[..., 2]-.5)+.010*ridge-.006*compact[..., 3]+snow_grain-.12*young_ice, .74, .92)
            ridge_keep = np.minimum(1, (lead+.010)*1200/max(footprint, 1))
            pressure = (1-smooth(lead+aa, lead+.010+aa, edge))*floe*ridge_keep
            marine_grain = .005*pattern(350, 665)+.014*pattern(90, 667)
            marine_light = np.clip(.88+.002*(compact[..., 2]-.5)-.002*compact[..., 3]+.028*pressure*np.clip(-compact[..., 3]*2, -1, 1)+marine_grain-.035*young_ice, .80, .93)
            compact_light = np.where(marine, marine_light, compact_light)
            local_light = local_light+(compact_light-local_light)*dense
            compact_rim = (1-smooth(lead+.006+aa, lead+.024+aa, edge))*floe*open_lead*.25
            marine_rim = (1-smooth(lead+.002+aa, lead+.008+aa, edge))*floe*open_lead*.06*ridge_keep
            compact_rim = np.where(marine, marine_rim, compact_rim)
            local_rim = local_rim+(compact_rim-local_rim)*dense
        pack = pack+(resolved-pack)*pack_keep
        brightness = F(.88)+(local_light-F(.88))*pack_keep
        rim = local_rim*pack_keep
        active = concentration > .002
        pack, rim = np.where(active, pack, 0), np.where(active, rim, 0)
    source = e.get('glacier', 0)*(.30+.70*smooth(1800, 30, depth))*(kind < 1.5)
    berg, berg_rim = np.zeros_like(pack), np.zeros_like(pack)
    berg_light = np.full_like(pack, .95)
    if np.any(source > .001):
        grouping = np.clip(.65+.23*pattern(2800, 647), .15, 1)
        density = source*grouping
        mean_berg = .040*density*(1-pack*.65)
        berg = mean_berg
        if footprint < 290:
            bend = pattern(380, 651)
            piece = _ice_cell(point+bend[..., None]*np.array([70, 35, -55], F), 850, seed+653)
            selected = smooth(.94-.42*density, .99-.42*density, piece[..., 2])*smooth(.001, .04, source)
            lobe = .022*pattern(68, 659)+.009*pattern(29, 661)
            cut = .065+.28*np.mod(piece[..., 2]*F(13.17), 1)
            field = piece[..., 0]-cut+lobe
            aa = max(.0015, footprint/850*.8)
            cap = smooth(-aa, aa, field)
            submerged = smooth(-.035-aa, -.012+aa, field)
            cast = smooth(-aa, aa, field+.12*piece[..., 3]-.006)*(1-cap)
            keep = 1-smooth(115, 290, footprint)
            berg = mean_berg+(cap*selected*(1-pack*.65)-mean_berg)*keep
            shadow = cast*selected*.42*keep*(1-pack)
            berg_rim = np.maximum(0, submerged-cap)*selected*.75*keep
            plateau = smooth(.006, .065, field)
            berg_light = F(.95)+(np.clip(.68+.29*plateau+.26*piece[..., 3]+.035*pattern(75, 663), .48, 1)-F(.95))*keep
        active = source > .001
        berg, shadow, berg_rim = (np.where(active, value, 0) for value in (berg, shadow, berg_rim))
    coverage = pack+(1-pack)*berg
    brightness = brightness+(berg_light-brightness)*berg/np.maximum(.001, coverage)
    return np.stack((coverage, brightness, np.maximum(rim, berg_rim), shadow), -1)


def water_color(depth, kind, cover, e, point, footprint, seed):
    sediment = smooth(.1, .9, e['sand'])*smooth(.1, 1, e['moisture'])
    bed = mix(mineral_color(e)*.65, [.20, .33, .24], sediment)
    shallow = mix(bed, [.12, .31, .32], np.full_like(depth, .6))
    # Bathymetric appearance retained from Render beyond optical shallows;
    # this does not claim visibility of a deep seafloor through clear water.
    marine = mix([.045, .16, .27], [.018, .065, .15], smooth(150, 3000, depth))
    deep = np.where((kind < 1.5)[..., None], marine, [.025, .12, .21])
    attenuation = np.where(kind < 1.5, .035, .15)
    water = mix(shallow, deep, 1-np.exp(-np.maximum(depth, 0)*attenuation*np.maximum(.08, cover*cover)))
    frozen = ice_material(depth, kind, e, point, footprint, seed)
    water = mix(water, [.12, .44, .48], frozen[..., 2]*.6)*(1-frozen[..., 3, None])
    ice = mix([.38, .60, .66], [.98, .99, .97], frozen[..., 1])
    return mix(water, ice, frozen[..., 0])


def colorize_surface(world, xs, ys, elevation, resolution, generation_settings, settings=None, *, polar=False, mode='render-v2', hydrology=None, lighting=None):
    """Retained material means with the V2 calendar and CPU relief lighting.

    Layer order follows the GPU mean compositor. This is not an integration
    of individual crowns, subpixel material normals or the fragment GGX BRDF.
    """
    from terrain_soil import sample_soil, seed_value, coordinates
    from terrain_koppen import seasonal_fields
    from terrain_lighting import relief_intensity
    from terrain_hydrology_render import ground_for_shading
    s = parse_settings(settings)
    if hydrology is not None:
        hydrology=np.asarray(hydrology,F).copy()
        valid=np.isfinite(hydrology).all(axis=-1) & (hydrology[...,3]>.5)
        hydrology[~valid]=0
    soil = sample_soil(world, xs, ys, generation_settings, polar=polar)
    ts0, ps, ls, tw0, pw, lw = seasonal_fields(world, xs, ys, generation_settings, polar=polar)
    point = coordinates(xs, ys, world.bounds, generation_settings.get('world_topology', 'sphere') != 'plane', polar)
    ground = ground_for_shading(elevation, hydrology)
    if hydrology is not None:
        from terrain_hydrology_render import DRY_GROUND_M
        dry_fraction = (hydrology[..., 3] > .5) & (hydrology[..., 2] >= 1.5) & (hydrology[..., 1] <= 0)
        ground = np.where(dry_fraction, np.maximum(elevation, DRY_GROUND_M), ground)
    gradient, tpi = terrain_derivatives(ground, resolution)
    ts, tw = ts0-ls*np.maximum(ground, 0), tw0-lw*np.maximum(ground, 0)
    bank = np.zeros_like(ground)
    step = resolution if np.isscalar(resolution) else max(resolution)
    if hydrology is not None:
        river = (hydrology[..., 2] > 2.5) & (hydrology[..., 3] > .5) & (hydrology[..., 1] > .05)
        if river.any():
            bank = 1-smooth(0, 65, distance_transform_edt(~river)*step)
            bank[river] = 0
    seed = seed_value(getattr(getattr(world, 'source', world), 'seed', 0))
    spherical = generation_settings.get('world_topology', 'sphere') != 'plane'
    land_point = point
    if spherical:
        xx, yy = np.meshgrid(xs, ys)
        x0, y0, x1, _ = world.bounds
        longitude = np.remainder((xx-x0)/(x1-x0), 1)*2*np.pi-np.pi
        radius = (x1-x0)/(2*np.pi)
        land_point = np.stack((radius*np.cos(longitude), radius*np.sin(longitude), yy-y0), -1).astype(F)
    e = evaluate(point, ground, gradient, tpi, ts, tw, ps, pw, soil[6], bank, s['season'], seed, step, land_point)
    # The marine evaluator retains canonical sphere coordinates and original
    # ecological inputs, so chart-local terrestrial detail cannot alter pack.
    marine = evaluate(point, ground, gradient, tpi, ts, tw, ps, pw, soil[6], bank, s['season'], seed, step)
    from terrain_glacier import sample_glacier_proximity
    e['glacier'] = sample_glacier_proximity(world,xs,ys,generation_settings,polar=polar)
    marine['glacier'] = e['glacier']
    from terrain_render import surface_material
    legacy = {}
    spacing = (step, step) if np.isscalar(resolution) else (resolution[1], resolution[0])
    ny, nx = np.gradient(point[..., 1], *spacing)
    north = np.stack((nx, ny), -1)
    north /= np.maximum(np.linalg.norm(north, axis=-1, keepdims=True), 1e-5)
    if not spherical:
        north.fill(0)
    surface_material(ground, gradient, tpi, point, north, step, soil,
                     (ts0, ps, ls, tw0, pw, lw), seed, s, capture=legacy, detail_point=land_point)
    bank_trees = bank*.5*smooth(5.5, 9, legacy['hot'])*(1-legacy['perennial'])
    e['woody'] = np.clip(legacy['woody']+bank_trees, 0, .985)
    e['grass'] = legacy['grass']*(1-e['snow'])
    e['rock'] = legacy['rock']
    e['sand'] = np.maximum(e['sand'], legacy['beach'])
    # The legacy beach capture includes its 0.8 colour blending strength.
    # Only its physical coverage excludes trees; low flats keep estuary woods.
    beach_coverage = np.clip(legacy['beach']/F(.8), 0, 1)
    for name in ('woody', 'grass', 'rock'):
        e[name] *= 1-beach_coverage
    cover = (ground < 0).astype(F)
    depth, kind = np.maximum(-elevation, 0), np.ones_like(ground)
    if hydrology is not None:
        valid = hydrology[..., 3] > .5
        inland = valid & (hydrology[..., 2] >= 1.5)
        cover = np.where(valid & (hydrology[..., 2] < .5), 0, cover)
        cover = np.where(inland, np.clip(hydrology[..., 1], 0, 1), cover)
        depth = np.where(inland, np.maximum(hydrology[..., 0]-elevation, 0), depth)
        kind = np.where(inland, hydrology[..., 2], kind)
    e['woody'] *= 1-cover
    e['grass'] *= 1-cover
    diagnostic = {'render-v2-substrate': 1, 'render-v2-moisture': 2, 'render-v2-vegetation': 3, 'render-v2-snow': 4}.get(mode, 0)
    rgb = colors(e, diagnostic, step)
    if diagnostic == 0:
        shown = appearance(e, step)
        # Retain one ground mean at every zoom. Mineral enrichment belongs
        # below coast sediment/snow, never a palette replacement at 6 m/px.
        flats = legacy['flats']*(1-smooth(240, 960, step))
        opened = ground_tint(legacy['open'], shown)
        opened = mix(opened, legacy['sediment']*.9, flats*.8)
        opened = mix(opened, legacy['sediment'], legacy['beach'])
        opened = mix(opened, [.88, .91, .91], e['snow'])
        leaf_on = dict(shown, leaf=np.ones_like(e['leaf']), snow=np.zeros_like(e['snow']))
        snow_free = dict(shown, snow=np.zeros_like(e['snow']))
        closed = legacy['canopy_leaf_on']*canopy_color(snow_free)/np.maximum(canopy_color(leaf_on), .02)
        closed = mix(closed, [.72, .79, .77], e['snow']*.62)
        # Render puts sediment over unresolved forest on low flats. Preserve
        # that mean without disallowing trees on all floodplains/estuaries.
        mean_closed = mix(closed, legacy['sediment']*.9, flats*.8*(1-e['snow']))
        mean_closed = mix(mean_closed, legacy['sediment'], legacy['beach']*(1-e['snow']))
        rgb = mix(opened, mean_closed, e['woody']*e['leaf'])
    if diagnostic == 0 and lighting is not None:
        rgb *= relief_intensity(ground, resolution, lighting)[..., None]
    if not np.any(cover > 0):
        return np.asarray(np.clip(rgb, 0, 1), F)
    water = water_color(depth, kind, cover, marine, point, step, seed)
    if diagnostic == 3:
        water = np.full_like(rgb, [0, 0, .13])
    if diagnostic == 4:
        water = mix([.08, .17, .24], [.85, .94, 1], ice_material(depth, kind, marine, point, step, seed)[..., 0])
    return np.asarray(np.clip(mix(rgb, water, cover), 0, 1), F)
