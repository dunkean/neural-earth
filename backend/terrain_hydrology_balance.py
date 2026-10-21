"""Climatological water balance, monthly routing and river properties.

Pure CPU NumPy: no Flask, torch or server imports. Water depths are mm over
the cell's own area, temperatures degrees C, volumes m3, discharge m3/s.
Month index 0 is January; a synthetic year has 365 days. Method note:
docs/methods/hydrology-balance.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BALANCE_VERSION = "hydro-balance-v1"
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], np.float64)
MONTH_SECONDS = MONTH_DAYS * 86400.0
MID_MONTH_DAY = np.cumsum(MONTH_DAYS) - MONTH_DAYS + 15  # FAO-56 representative days
# Orogen's "summer" fields are northern: temperature proxies JJA/DJF (harmonic
# peak in July), rain totals for the northern half-years Apr-Sep / Oct-Mar.
NORTH_SUMMER_HALF = np.arange(12) // 3 % 3 != 0
PEAK_MONTH = 6
BIO_SAMPLE_CORRECTION = np.sqrt(12 / 11)  # dismo::biovars sample std
MAX_MONTHLY_CELLS = 1 << 20
UNITS = {"water": "mm per cell area", "temperature": "degC", "volume": "m3",
         "discharge": "m3/s", "length": "m", "slope": "m/m"}


@dataclass(frozen=True)
class BalanceParams:
    snow_all_c: float = -1.0          # monthly mean at/below: all snow
    rain_all_c: float = 3.0           # monthly mean at/above: all rain
    degree_day_mm: float = 3.0        # mm / degC / day above 0 C (Hock 2003 snow range)
    snow_cap_mm: float = 2000.0       # perennial snow SWE; excess leaves as ice-discharge proxy
    soil_capacity_mm: float = 150.0   # Thornthwaite-Mather available water
    recharge_fraction: float = 0.35   # share of soil surplus entering groundwater
    groundwater_days: float = 90.0    # linear-reservoir recession time constant
    tolerance_mm: float = 0.01        # periodic-state tolerance, summed over stores
    max_years: int = 40


def _arr(value):
    return None if value is None else np.asarray(value, np.float64)


def extraterrestrial_radiation(latitude_deg, month):
    """FAO-56 eq. 21 daily Ra (MJ m-2 day-1) on the representative mid-month day."""
    phi = np.radians(np.clip(latitude_deg, -89.9, 89.9))
    angle = 2 * np.pi * MID_MONTH_DAY[month] / 365
    dr, delta = 1 + 0.033 * np.cos(angle), 0.409 * np.sin(angle - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return np.maximum(24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) +
                                                       np.cos(phi) * np.cos(delta) * np.sin(ws)), 0)


def oudin_pet_mm(temp_c, latitude_deg, month):
    """Oudin et al. (2005) potential evapotranspiration in mm/month."""
    # PE = Re/(lambda rho) (T+5)/100 with lambda 2.45 MJ/kg, rho 1000 kg/m3.
    ra = extraterrestrial_radiation(latitude_deg, month)
    return ra * np.maximum(np.asarray(temp_c) + 5, 0) / 245 * MONTH_DAYS[month]


def _seasonal_inputs(precip_mm_year, temp_c, latitude_deg, bio4, temp_summer_c, temp_winter_c,
                     precip_summer_mm, precip_winter_mm, bio15, wet_half, dtype=np.float64):
    """Flat (5, n) mean temperature, signed July amplitude, northern half totals, latitude."""
    fields = dict(p=_arr(precip_mm_year), t=_arr(temp_c), lat=_arr(latitude_deg), bio4=_arr(bio4),
                  ts=_arr(temp_summer_c), tw=_arr(temp_winter_c), ps=_arr(precip_summer_mm),
                  pw=_arr(precip_winter_mm), bio15=_arr(bio15))
    if fields["lat"] is None:
        raise ValueError("latitude_deg is required for radiation and hemisphere phase")
    if (fields["ts"] is None) != (fields["tw"] is None) or (fields["ps"] is None) != (fields["pw"] is None):
        raise ValueError("seasonal Orogen fields must be given in summer/winter pairs")
    shape = np.broadcast_shapes(*(v.shape for v in fields.values() if v is not None))
    flat = {k: np.broadcast_to(v, shape).reshape(-1) for k, v in fields.items() if v is not None}
    if not all(np.isfinite(v).all() for v in flat.values()):
        raise ValueError("climate inputs must be finite")
    north = flat["lat"] >= 0
    meta = {}
    if "ts" in flat:
        mean = flat["t"] if "t" in flat else (flat["ts"] + flat["tw"]) * .5
        amp, meta["temperature_phase"] = (flat["ts"] - flat["tw"]) * .5, "orogen JJA/DJF proxies, July peak"
    elif "t" in flat:
        mean = flat["t"]
        if "bio4" in flat:
            # BIO4 = 100 * sample std of 12 harmonic months = 100 * A/sqrt(2) * sqrt(12/11).
            amp = flat["bio4"] / 100 * np.sqrt(2) / BIO_SAMPLE_CORRECTION * np.where(north, 1, -1)
            meta["temperature_phase"] = "BIO4 harmonic, convention: local summer peak July N / January S"
        else:
            amp, meta["temperature_phase"] = np.zeros_like(mean), "none (constant temperature)"
    else:
        raise ValueError("temp_c or temp_summer_c/temp_winter_c is required")
    if "ps" in flat:
        ps, pw = flat["ps"], flat["pw"]
        meta["precipitation_phase"] = "orogen half-year totals, equal months within Apr-Sep / Oct-Mar"
    elif "p" in flat and "bio15" in flat and wet_half is not None:
        if wet_half not in ("summer", "winter"):
            raise ValueError("wet_half must be 'summer' or 'winter' (local hemisphere)")
        p = flat["p"]
        diff = np.clip(flat["bio15"] / (100 * BIO_SAMPLE_CORRECTION) * (p + 12), 0, p)
        wet, dry = (p + diff) * .5, (p - diff) * .5
        north_wet = north == (wet_half == "summer")
        ps, pw = np.where(north_wet, wet, dry), np.where(north_wet, dry, wet)
        meta["precipitation_phase"] = f"BIO15 inverse of the Orogen half-year encoding, convention: local {wet_half} wet"
    elif "p" in flat:
        ps = pw = flat["p"] * .5
        meta["precipitation_phase"] = "uniform (BIO15 carries no monthly phase)"
    else:
        raise ValueError("precip_mm_year or precip_summer_mm/precip_winter_mm is required")
    if (ps < 0).any() or (pw < 0).any():
        raise ValueError("precipitation cannot be negative")
    return shape, np.stack(np.broadcast_arrays(mean, amp, ps, pw, flat["lat"]), dtype=dtype), meta


def _forcing(inputs, p):
    """(5, 12, n) temperature, snowfall, rain, melt capacity and Oudin PET for a chunk."""
    mean, amp, ps, pw, lat = inputs
    months = np.arange(12)[:, None]
    temp = mean + amp * np.cos(2 * np.pi * (months - PEAK_MONTH) / 12)
    precip = np.where(NORTH_SUMMER_HALF[:, None], ps / 6, pw / 6)
    snowfall = precip * np.clip((p.rain_all_c - temp) / (p.rain_all_c - p.snow_all_c), 0, 1)
    lats, inverse = np.unique(lat, return_inverse=True)  # radiation depends on latitude only
    ra = extraterrestrial_radiation(lats, months)[:, inverse]
    pet = ra * np.maximum(temp + 5, 0) / 245 * MONTH_DAYS[:, None]
    melt = p.degree_day_mm * np.maximum(temp, 0) * MONTH_DAYS[:, None]
    return np.stack((temp, snowfall, precip - snowfall, melt, pet))


def _run_year(state, forcing, p, record=None):
    """Advance (snow, soil, groundwater) one year; returns end state, totals, min snow."""
    snow, soil, gw = (s.copy() for s in state)
    totals = np.zeros((3, snow.size))  # precipitation, evapotranspiration, runoff
    snow_min = np.full_like(snow, np.inf)
    for m in range(12):
        temp, snowfall, rain, melt_capacity, pet = forcing[:, m]
        snow += snowfall
        melt = np.minimum(snow, melt_capacity)
        snow -= melt
        overflow = np.maximum(snow - p.snow_cap_mm, 0)
        snow -= overflow
        np.minimum(snow_min, snow, out=snow_min)
        liquid = rain + melt + overflow
        # Thornthwaite-Mather: a surplus month adds liquid - PET; a deficit
        # month depletes soil by exp(-deficit / capacity). AET <= PET.
        deficit = pet - liquid
        new_soil = soil * np.exp(np.maximum(deficit, 0) / -p.soil_capacity_mm) - np.minimum(deficit, 0)
        aet = liquid + soil - new_soil
        surplus = np.maximum(new_soil - p.soil_capacity_mm, 0)
        soil = new_soil - surplus
        recharge = surplus * p.recharge_fraction
        gw += recharge
        baseflow = gw * -np.expm1(-MONTH_DAYS[m] / p.groundwater_days)
        gw -= baseflow
        runoff = surplus - recharge + baseflow
        totals[0] += snowfall + rain
        totals[1] += aet
        totals[2] += runoff
        if record is not None:
            for name, value in (("temperature_c", temp), ("precipitation_mm", snowfall + rain),
                                ("snowfall_mm", snowfall), ("melt_mm", melt + overflow),
                                ("potential_evaporation_mm", pet), ("evapotranspiration_mm", aet),
                                ("runoff_mm", runoff), ("baseflow_mm", baseflow),
                                ("snow_mm", snow), ("soil_mm", soil), ("groundwater_mm", gw)):
                record.setdefault(name, []).append(value.copy())
    return (snow, soil, gw), totals, snow_min


def _periodic_state(forcing, p):
    """Deterministic per-cell warmup to the climatological periodic start state.

    Each cell's trajectory depends only on its own forcing, so results do not
    depend on chunking. Accelerations: perennially growing snow jumps to its
    cap; groundwater uses the exact fixed point of its linear annual map; soil
    uses the secant fixed point of its (piecewise affine) annual map.
    Also returns the totals and store changes of each cell's final year.
    """
    n = forcing.shape[-1]
    state, totals, final_change = np.zeros((3, n), forcing.dtype), np.zeros((3, n)), np.zeros((3, n))
    years, done = np.zeros(n, np.int16), np.zeros(n, bool)
    prev_soil_change = np.zeros(n, forcing.dtype)
    gw_decay = np.exp(-365 / p.groundwater_days)
    idx = np.arange(n)
    for year in range(1, p.max_years + 1):
        start = state[:, idx]
        end, totals[:, idx], snow_min = _run_year(start, forcing if idx.size == n else forcing[..., idx], p)
        end = np.stack(end)
        change = end - start
        final_change[:, idx] = change
        years[idx] = year
        ok = np.abs(change).sum(0) <= p.tolerance_mm
        done[idx[ok]] = True
        nxt = end.copy()
        nxt[0] = np.where((snow_min > 0) & (change[0] > 0), p.snow_cap_mm, end[0])
        prev = prev_soil_change[idx]
        secant = (prev * change[1] > 0) & (np.abs(change[1]) < np.abs(prev))
        alpha = np.where(secant, change[1] / np.where(secant, prev, 1), 0)
        nxt[1] = np.clip((end[1] - alpha * start[1]) / (1 - alpha), 0, p.soil_capacity_mm)
        nxt[2] = (end[2] - gw_decay * start[2]) / (1 - gw_decay)
        keep = ~ok
        state[:, idx[keep]] = nxt[:, keep]
        prev_soil_change[idx] = change[1]
        idx = idx[keep]
        if not idx.size:
            break
    return state, done, years, totals, final_change


def _balance(args, params, chunk_cells, monthly):
    p = params or BalanceParams()
    # Float32 stores for the global pass halve memory traffic; regions keep float64.
    dtype = np.float64 if monthly else np.float32
    shape, inputs, meta = _seasonal_inputs(*args, dtype=dtype)
    n = inputs.shape[1]
    if monthly and n > MAX_MONTHLY_CELLS:
        raise ValueError("monthly_water_balance is for regions/reach units; use annual_water_balance")
    names = (("precipitation_mm_year", "evapotranspiration_mm_year", "runoff_mm_year",
              "potential_evaporation_mm_year", "storage_change_mm", "residual_mm") if not monthly else ())
    out = {k: np.empty(n, np.float32) for k in names}
    out.update(converged=np.empty(n, bool), warmup_years=np.empty(n, np.int16))
    rec_all = {}
    for s in range(0, n, chunk_cells):
        sl = slice(s, min(n, s + chunk_cells))
        forcing = _forcing(inputs[:, sl], p).astype(dtype, copy=False)
        start, out["converged"][sl], out["warmup_years"][sl], totals, change = _periodic_state(forcing, p)
        if monthly:
            record = {}
            _run_year(start, forcing, p, record)
            for k, v in record.items():
                rec_all.setdefault(k, []).append(np.stack(v))
            rec_all.setdefault("initial_storage_mm", []).append(start)
            continue
        out["precipitation_mm_year"][sl], out["evapotranspiration_mm_year"][sl], out["runoff_mm_year"][sl] = totals
        out["potential_evaporation_mm_year"][sl] = forcing[4].sum(0)
        out["storage_change_mm"][sl] = change.sum(0)
        out["residual_mm"][sl] = totals[0] - totals[1] - totals[2] - change.sum(0)
    if monthly:
        out.update({k: np.concatenate(v, axis=1) for k, v in rec_all.items()})
        storage = out["snow_mm"] + out["soil_mm"] + out["groundwater_mm"]
        before = np.concatenate((out["initial_storage_mm"].sum(0)[None], storage[:-1]))
        out["storage_change_mm"] = storage - before
        out["residual_mm"] = (out["precipitation_mm"] - out["evapotranspiration_mm"] -
                              out["runoff_mm"] - out["storage_change_mm"])
    result = {k: (v.reshape(v.shape[:-1] + shape) if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    result["metadata"] = dict(meta, version=BALANCE_VERSION, units=UNITS, params=asdict(p),
                              months_start="January", year_days=365)
    return result


def annual_water_balance(precip_mm_year, temp_c, *, latitude_deg, bio4=None,
                         temp_summer_c=None, temp_winter_c=None, precip_summer_mm=None,
                         precip_winter_mm=None, bio15=None, wet_half=None, params=None,
                         chunk_cells=65536):
    """Periodic-year totals per cell, arrays shaped like the broadcast inputs.

    Returns float32 mm/year ``runoff_mm_year`` (already net of land
    evapotranspiration), ``evapotranspiration_mm_year``,
    ``potential_evaporation_mm_year`` (Oudin; open-water loss proxy),
    ``precipitation_mm_year``, periodic ``storage_change_mm``, numerical
    ``residual_mm``, plus ``converged``, ``warmup_years`` and ``metadata``.
    Memory is bounded by ``chunk_cells``; no 12-month global tensors are kept.
    """
    args = (precip_mm_year, temp_c, latitude_deg, bio4, temp_summer_c, temp_winter_c,
            precip_summer_mm, precip_winter_mm, bio15, wet_half)
    return _balance(args, params, chunk_cells, monthly=False)


def monthly_water_balance(precip_mm_year, temp_c, *, latitude_deg, bio4=None,
                          temp_summer_c=None, temp_winter_c=None, precip_summer_mm=None,
                          precip_winter_mm=None, bio15=None, wet_half=None, params=None,
                          chunk_cells=65536):
    """Periodic climatological year, float64 arrays shaped (12, *input_shape).

    Fluxes (mm/month): temperature_c, precipitation_mm, snowfall_mm, melt_mm
    (includes snow-cap overflow), potential_evaporation_mm,
    evapotranspiration_mm, runoff_mm, baseflow_mm. End-of-month stores (mm):
    snow_mm, soil_mm, groundwater_mm; storage_change_mm and residual_mm per
    month. ``initial_storage_mm`` is (3, *shape) for snow, soil, groundwater.
    Limited to MAX_MONTHLY_CELLS: intended for region/reach units.
    """
    args = (precip_mm_year, temp_c, latitude_deg, bio4, temp_summer_c, temp_winter_c,
            precip_summer_mm, precip_winter_mm, bio15, wet_half)
    return _balance(args, params, chunk_cells, monthly=True)


def topological_levels(downstream):
    """Kahn levels of an acyclic downstream graph (-1 terminal), upstream first."""
    downstream = np.asarray(downstream, np.int64)
    n = downstream.size
    has = downstream >= 0
    pending = np.bincount(downstream[has], minlength=n)
    level, levels, seen = np.flatnonzero(pending == 0), [], 0
    while level.size:
        levels.append(level)
        seen += level.size
        down = downstream[level]
        down = down[down >= 0]
        np.subtract.at(pending, down, 1)
        level = np.unique(down[pending[down] == 0])
    if seen != n:
        raise ValueError("downstream graph contains a cycle")
    return levels


def route_monthly(downstream, lateral_m3, *, loss_m3=None, levels=None, conserve_annual_loss=False):
    """Conservatively accumulate volumes down reaches or cells.

    ``downstream`` (N,) int, -1 terminal. ``lateral_m3`` (N,) or (N, M) is
    the non-negative volume generated by each node's disjoint lateral
    catchment per period (e.g. runoff_mm / 1000 * area_m2). Optional
    ``loss_m3`` is a potential transmission/evaporation loss; the actual loss
    never exceeds available water. Flow-through only: no lake storage.
    Returns inflow/lateral/loss/outflow volumes, ``terminal_m3`` per period
    and the global ``residual_m3`` (lateral - terminal - loss).
    """
    downstream = np.asarray(downstream, np.int64)
    lateral = np.asarray(lateral_m3, np.float64)
    if (lateral < 0).any():
        raise ValueError("lateral volumes must be non-negative")
    potential = np.zeros(lateral.shape) if loss_m3 is None else np.asarray(loss_m3, np.float64)
    potential = np.broadcast_to(potential.reshape(potential.shape + (1,) * (lateral.ndim - potential.ndim)),
                                lateral.shape)  # per-node losses apply to every period
    inflow, loss, outflow = (np.zeros_like(lateral) for _ in range(3))
    for level in levels or topological_levels(downstream):
        available = inflow[level] + lateral[level]
        loss[level] = np.clip(potential[level], 0, available)
        if conserve_annual_loss and lateral.ndim == 2:
            # Availability-constrained allocation of a fixed annual loss.
            # This is a seasonal proxy, not an unmodelled lake reservoir:
            # evaporation shifts to months with water, and creates no flow.
            target = np.minimum(np.maximum(potential[level], 0).sum(1), available.sum(1))
            allocated = loss[level].copy()
            for _ in range(lateral.shape[1]):
                remaining = np.maximum(target-allocated.sum(1), 0)
                if np.max(remaining, initial=0.) < 1e-10:
                    break
                room = np.maximum(available-allocated, 0)
                weight = np.where(room > 0, np.maximum(potential[level], target[:, None]*1e-12), 0)
                share = np.divide(weight, weight.sum(1, keepdims=True),
                                  out=np.zeros_like(weight), where=weight.sum(1, keepdims=True) > 0)
                allocated += np.minimum(room, remaining[:, None]*share)
            loss[level] = allocated
        outflow[level] = available - loss[level]
        down = downstream[level]
        has = down >= 0
        np.add.at(inflow, down[has], outflow[level][has])
    terminal = outflow[downstream < 0].sum(0)
    return dict(inflow_m3=inflow, lateral_m3=lateral, loss_m3=loss, outflow_m3=outflow,
                terminal_m3=terminal, residual_m3=lateral.sum(0) - terminal - loss.sum(0))


def flow_statistics(q_m3s, *, dry_threshold_m3s=0.01):
    """Climatological summaries of (..., 12) monthly discharge.

    ``min_month_m3s`` is the lowest of the 12 synthetic monthly means, not a
    measured Q95/QMNA5. ``dry_months`` counts months at/below the threshold.
    """
    q = np.asarray(q_m3s, np.float64)
    return dict(mean_m3s=(q * MONTH_SECONDS).sum(-1) / MONTH_SECONDS.sum(),
                min_month_m3s=q.min(-1), min_month=q.argmin(-1), max_month_m3s=q.max(-1),
                dry_months=(q <= dry_threshold_m3s).sum(-1))


def channel_hydraulics(q_m3s, slope, *, manning_n=0.035, min_slope=1e-5):
    """Wet width, depth and velocity for (N, 12) monthly discharge.

    Mean-flow width follows Moody & Troutman (2002), 7.2 Q^0.5; monthly
    width scales at-a-station with exponent 0.26 (Leopold & Maddock 1953).
    Depth is Manning normal flow in a wide channel where slope >= min_slope,
    otherwise the hydraulic-geometry prior 0.27 Q^0.39 (scaled with 0.40).
    """
    q = np.maximum(np.asarray(q_m3s, np.float64), 0)
    slope = np.asarray(slope, np.float64)[..., None]
    q_mean = np.maximum((q * MONTH_SECONDS).sum(-1, keepdims=True) / MONTH_SECONDS.sum(), 1e-12)
    ratio = q / q_mean
    width = 7.2 * np.sqrt(q_mean) * ratio ** 0.26
    with np.errstate(divide="ignore", invalid="ignore"):
        manning = (q * manning_n / (width * np.sqrt(np.maximum(slope, min_slope)))) ** 0.6
        depth = np.where(slope >= min_slope, manning, 0.27 * q_mean ** 0.39 * ratio ** 0.40)
        depth = np.where(q > 0, depth, 0)
        velocity = np.where(q > 0, q / (width * depth), 0)
    return np.where(q > 0, width, 0), depth, velocity


VESSELS = (  # explicit worldbuilding heuristics, least to most demanding
    dict(name="canoe", min_depth_m=0.4, min_width_m=4.0, max_velocity_m_s=1.5, max_slope=3e-3),
    dict(name="riverboat", min_depth_m=1.5, min_width_m=20.0, max_velocity_m_s=2.0, max_slope=5e-4),
    dict(name="ship", min_depth_m=4.0, min_width_m=80.0, max_velocity_m_s=2.5, max_slope=1.5e-4),
)
REASONS = ("ok", "dry", "depth", "width", "current", "slope", "obstacle", "no_outlet")


def navigability(downstream, length_m, slope, q_m3s, *, outlet_navigable, obstacle=None,
                 vessels=VESSELS, manning_n=0.035, levels=None):
    """Seasonal route navigability of reaches for each vessel class.

    A reach is navigable in a month if its own section passes the vessel
    thresholds *and* the route continues to a navigable outlet: either the
    reach itself ends at sea/lake/port (``outlet_navigable``) or its
    downstream reach is navigable that month. ``obstacle`` (N,) or (N, 12)
    marks falls, dams or unrunnable rapids supplied by geometry.
    Arrays are (V, N, 12): ``navigable``, ``limit_reach`` (first failing
    reach on the route, -1 if none), ``limit_reason`` (index into REASONS),
    ``route_length_m`` (navigable distance to the outlet).
    """
    downstream = np.asarray(downstream, np.int64)
    length = np.asarray(length_m, np.float64)
    slope = np.asarray(slope, np.float64)
    q = np.asarray(q_m3s, np.float64)
    n = downstream.size
    width, depth, velocity = channel_hydraulics(q, slope, manning_n=manning_n)
    blocked = np.zeros((n, 12), bool) if obstacle is None else np.broadcast_to(
        np.asarray(obstacle, bool).reshape(n, -1), (n, 12))
    outlet = np.broadcast_to(np.asarray(outlet_navigable, bool).reshape(n, -1), (n, 12))
    levels = levels or topological_levels(downstream)
    shape = (len(vessels), n, 12)
    nav, limit = np.zeros(shape, bool), np.full(shape, -1, np.int32)
    reason, route = np.zeros(shape, np.int8), np.zeros(shape)
    for v, vessel in enumerate(vessels):
        local = np.zeros((n, 12), np.int8)  # first failing local reason, 0 = ok
        steep = np.broadcast_to(slope[:, None] > vessel["max_slope"], (n, 12))
        for code, fail in ((4, velocity > vessel["max_velocity_m_s"]), (3, width < vessel["min_width_m"]),
                           (2, depth < vessel["min_depth_m"]), (5, steep), (6, blocked), (1, q <= 0)):
            local[fail] = code  # later checks win: dry, obstacle, slope, depth, width, current
        for level in reversed(levels):  # downstream first
            down = downstream[level]
            terminal = outlet[level] | (down < 0)[:, None]
            dn = np.maximum(down, 0)
            own = local[level]
            reach_ok = own == 0
            down_ok = np.where(terminal, outlet[level], nav[v, dn])
            nav[v, level] = reach_ok & down_ok
            route[v, level] = np.where(nav[v, level], length[level][:, None] +
                                       np.where(outlet[level], 0, route[v, dn]), 0)
            own_fail = ~reach_ok
            no_outlet = reach_ok & terminal & ~outlet[level]
            inherited = reach_ok & ~terminal
            limit[v, level] = np.where(own_fail | no_outlet, level[:, None],
                                       np.where(inherited, limit[v, dn], -1))
            reason[v, level] = np.where(own_fail, own, np.where(no_outlet, 7,
                                        np.where(inherited, reason[v, dn], 0)))
    months = nav.sum(-1)
    capable = np.arange(len(vessels))[:, None]
    return dict(vessels=tuple(x["name"] for x in vessels), reasons=REASONS, width_m=width,
                depth_m=depth, velocity_m_s=velocity, navigable=nav, limit_reach=limit,
                limit_reason=reason, route_length_m=route, months_navigable=months,
                year_round_class=np.where(months == 12, capable, -1).max(0),
                seasonal_class=np.where(months > 0, capable, -1).max(0))
