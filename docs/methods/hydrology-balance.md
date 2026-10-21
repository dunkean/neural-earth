# Hydrology balance, monthly routing and river properties

Module: `backend/terrain_hydrology_balance.py` (`BALANCE_VERSION = "hydro-balance-v1"`). Pure CPU NumPy; no Flask, torch or server import. Tests: `tests/python/test_terrain_hydrology_balance.py`. Design context: [multiscale hydrology](../design/multiscale-hydrology.md), section 4.

This is an uncalibrated, explicitly parameterised climatological model for worldbuilding. It is not HBV, GR4J or a calibrated hydrological model, and it computes no storms, floods, return periods or measured low-flow statistics.

## Inputs and units

All inputs broadcast together. Latitude is required: it sets radiation and hemisphere phase. For an atlas, a `(H, 1)` column is enough.

| Argument | Unit | Source |
| --- | --- | --- |
| `precip_mm_year` | mm/year | BIO12 |
| `temp_c` | degC at the cell's elevation | BIO1 (conditioning already applies the lapse rate) |
| `bio4` | 100 x sample std of 12 monthly temperatures | BIO4 |
| `bio15` | percent, monthly CV of (rain + 1 mm), sample std | BIO15 |
| `temp_summer_c`, `temp_winter_c` | degC | Orogen northern JJA / DJF proxies (`encoded * 90 - 45`) |
| `precip_summer_mm`, `precip_winter_mm` | mm per half-year | Orogen northern Apr-Sep / Oct-Mar totals (`encoded * 1000`) |

These units follow `terrain_conditioning.orogen_bioclim` and `CHANNEL_NAMES`. Outputs are mm over the cell's own area. Volumes are m3 and discharge m3/s. Month 0 is January. The synthetic year has 365 days, so `MONTH_SECONDS` are the seconds in each calendar month.

## Seasonal forcing (no invented phase)

- **Temperature.** `T_m = mean + A cos(2 pi (m - 6) / 12)`, with July at the peak. With Orogen pairs, `A = (T_summer - T_winter) / 2`, which reproduces both proxies exactly in July and January. If `temp_c` is also given, it is used as the mean. With BIO4 only, `A = BIO4 / 100 * sqrt(2) / sqrt(12/11)`, the exact inverse of the harmonic BIO4 in `orogen_bioclim`. Its sign follows a labelled hemisphere convention: warm peak in July for latitude >= 0, January otherwise. Without BIO4, temperature is constant through the year.
- **Precipitation.** Orogen half-year totals are split equally over the six months of each northern half-year. This is the same minimum-CV allocation `orogen_bioclim` assumes, so the half-year totals are exact. With BIO12 and BIO15 only, monthly rain is **uniform** by default, because BIO15 carries no phase. Passing `wet_half='summer'|'winter'` (meaning the local hemisphere season) applies the exact inverse of the project's Orogen BIO15 encoding, `|P_s - P_w| = BIO15 / (100 sqrt(12/11)) (P + 12)`, to the caller's chosen wet half. Real WorldClim BIO15 does not encode half-years, so this is a convention, not recovered weather.
- `metadata['temperature_phase']` and `metadata['precipitation_phase']` record the rule actually used.

## Monthly stores, per cell, in mm

The order within each month:

1. **Snow.** Rain/snow partition is linear in monthly mean temperature between `snow_all_c = -1` and `rain_all_c = 3` degC. Temperature-index melt is `min(SWE, DDF * max(T, 0) * days)` with `DDF = 3 mm/degC/day`, a snow value within the ranges reviewed by [Hock, 2003](https://doi.org/10.1016/S0022-1694(03)00257-9). SWE above `snow_cap_mm = 2000` leaves as liquid input. This overflow is an explicit steady-state ice-discharge proxy, not glacier mass balance; without it, perennially accumulating cells never reach a periodic state.
2. **Potential evapotranspiration.** [Oudin et al., 2005](https://doi.org/10.1016/j.jhydrol.2004.08.026): `PE = Re / (lambda rho) * (T + 5) / 100` when `T > -5`, otherwise 0. Here `lambda = 2.45 MJ/kg` and `rho = 1000 kg/m3`, giving `PE[mm/day] = Re (T + 5) / 245`. `Re` is the FAO-56 eq. 21 extraterrestrial radiation on the representative mid-month day ([Allen et al., 1998](https://www.fao.org/4/x0490e/x0490e07.htm)). It uses monthly mean temperature, not daily values.
3. **Soil.** A Thornthwaite-Mather monthly bucket with capacity `soil_capacity_mm = 150`. When liquid input is at least PE, AET = PE and the soil gains `liquid - PE`; anything above capacity is surplus. In a deficit month, `soil *= exp(-(PE - liquid) / capacity)` and `AET = liquid + depletion <= PE`.
4. **Groundwater.** `recharge_fraction = 0.35` of the surplus recharges a linear reservoir. Monthly baseflow is `G (1 - exp(-days / 90))`.
5. **Outputs.** Runoff is `surplus - recharge + baseflow`. Runoff is exactly zero in any cell that never produces a soil surplus, so there is no positive floor. ET is land actual evapotranspiration, already removed from runoff.

The parameters live in the frozen `BalanceParams` dataclass, and `metadata['params']` records them. They are plausible defaults, not calibrated against observations.

## Periodic warmup

Every cell starts with empty stores and repeats the synthetic year until `sum |store_end - store_start| <= tolerance_mm` (0.01 mm), or until `max_years` (40). Three deterministic accelerations help:

- Snow that never empties and grows over the year jumps to its cap.
- Groundwater takes the exact fixed point of its linear annual map: `G* = (G_end - a G_start) / (1 - a)`, with `a = exp(-365 / tau)`.
- Soil uses the secant fixed point of its annual map, which is piecewise affine. It is only applied when two successive changes contract with the same sign.

Each cell's trajectory depends only on its own forcing, so results are independent of chunking; a test checks `chunk_cells=1` against the default. `converged` and `warmup_years` are returned per cell. The annual totals come from the verified periodic year, `storage_change_mm` is its residual store drift, and `residual_mm = P - ET - R - dS` is the numerical closure.

## API

- `annual_water_balance(precip_mm_year, temp_c, *, latitude_deg, ...)` returns float32 arrays with the broadcast input shape: `runoff_mm_year`, `evapotranspiration_mm_year`, `potential_evaporation_mm_year`, `precipitation_mm_year`, `storage_change_mm`, `residual_mm`, plus `converged`, `warmup_years` and `metadata`. It processes `chunk_cells` cells at a time (65536 by default) with float32 stores and float64 totals, and keeps no global 12-month tensors.
- `monthly_water_balance(...)` returns float64 `(12, *shape)` arrays: `temperature_c`, `precipitation_mm`, `snowfall_mm`, `melt_mm` (including cap overflow), `potential_evaporation_mm`, `evapotranspiration_mm`, `runoff_mm`, `baseflow_mm`, and end-of-month `snow_mm`, `soil_mm` and `groundwater_mm`. It also returns per-month `storage_change_mm` and `residual_mm`, and `initial_storage_mm` with shape `(3, *shape)`. It is limited to `MAX_MONTHLY_CELLS` (2^20) because it is meant for reach and region units.
- `route_monthly(downstream, lateral_m3, *, loss_m3=None, levels=None)` accumulates volumes conservatively over any acyclic graph, either reaches or cells, with -1 as the terminal. `lateral_m3` has shape `(N,)` or `(N, M)` and holds the volume of each node's disjoint lateral catchment, for example `runoff_mm / 1000 * area_m2`. The optional potential loss (transmission or channel evaporation) is capped by the water available at the node, so arid reaches can lose all their flow. It returns per-node inflow, lateral, loss and outflow, `terminal_m3`, and `residual_m3 = lateral - terminal - loss`. Monthly discharge is `outflow_m3 / MONTH_SECONDS`. Processing goes level by level through Kahn levels from `topological_levels`, and the result does not depend on node labels.
- `flow_statistics(q)` takes `q` with shape `(..., 12)` and returns the time-weighted mean, the minimum and maximum climatological monthly flow with the month index, and `dry_months` (q <= 0.01 m3/s by default). The minimum is the lowest of 12 synthetic monthly means, **not** Q95 or QMNA5.
- `channel_hydraulics(q, slope)` returns wet width, depth and velocity with shape `(N, 12)`. Mean-flow width is `7.2 Q^0.5`, from the global hydraulic-geometry family of [Moody and Troutman (2002)](https://pubs.usgs.gov/publication/70024149) used by [Andreadis et al. (2013)](https://doi.org/10.1002/wrcr.20440). The explicit depth prior `0.27 Q^0.39` follows the practical coefficients documented by [BasinMaker](https://basinmaker.readthedocs.io/en/latest/basinmaker_tools.html); [GLOBAL-FATE (Font et al., 2019)](https://gmd.copernicus.org/articles/12/5213/2019/gmd-12-5213-2019.html) also records that exponent. Here Q is explicitly m³/s. These defaults applied to annual mean discharge are a procedural prior, not a verified bankfull or site calibration. Monthly values follow at-a-station exponents of 0.26 for width and 0.40 for depth, after Leopold and Maddock (1953). Depth is Manning normal flow in a wide channel, `d = (Q n / (w sqrt S))^(3/5)` with `n = 0.035`, for `S >= 1e-5`. Manning is never inverted at zero slope; flatter reaches use the prior `0.27 Q^0.39`. This is a global prior, not a regional calibration, and it is not a backwater or estuary solver.
- `navigability(downstream, length_m, slope, q_m3s, *, outlet_navigable, obstacle=None)` works on reaches for each vessel in `VESSELS` (canoe, riverboat, ship; explicit heuristic thresholds for minimum depth, minimum width, maximum velocity and maximum slope).
  - A reach is navigable in a month only if its own section passes the thresholds **and** the route continues downstream to a reach flagged `outlet_navigable` (sea, lake or port). A short deep segment above a fall therefore does not count.
  - Outputs, each with shape `(V, N, 12)`: `navigable`, `limit_reach` (the first failing reach on the route), `limit_reason` (an index into `REASONS`: dry, depth, width, current, slope, obstacle or no_outlet), and `route_length_m`.
  - Per reach it also gives `year_round_class` and `seasonal_class`.
  - Falls, dams and rapids come from geometry through `obstacle`; the slope threshold is only a rapids proxy.

## Integration notes for the core build

- Pass `runoff_mm_year` as core `runoff_mm_year`. It is already net of land ET, so core must not subtract land evaporation again.
- For core lake and open-water losses (`evaporation_mm_year`), use `potential_evaporation_mm_year`. Oudin PE is only a proxy for open-water evaporation.
- Monthly unit budgets: sum `runoff_mm * area_m2 / 1000` over each unit's disjoint cells, then call `route_monthly` on the reach graph. Child partitions of a parent's lateral volume keep the parent totals.

## Evidence (2026-10-21, local machine, other workers active)

- `.venv/Scripts/python.exe -m unittest tests.python.test_terrain_hydrology_balance -v`: 6 tests OK in 0.13 s. They cover:
  - zero runoff in arid cells, annual closure and chunk invariance;
  - exact Orogen half-year totals and July/January phase, BIO4/BIO15 consistency with `orogen_bioclim`, and BIO-only reconstruction under the labelled convention;
  - delayed snow release, with the pack peaking in March and runoff peaking in April or May;
  - periodic storage within tolerance, numerical residual below 1e-9 mm, and capped perennial ice;
  - conservative lossy routing that is label-independent and rejects cycles;
  - a navigation bottleneck at falls, plus seasonal ship access limited by depth.
- Synthetic 2048 x 1024 annual pass with BIO1, BIO4 and BIO12 and `(H, 1)` latitude:
  - 6.0 s, with a tracemalloc peak of about 180 MiB including inputs.
  - Every cell converged, in 3.04 warmup years on average and 31 at most (slow perennial cells).
  - Maximum `|residual|` was 1.7e-3 mm in float32, at most 1.9e-4 relative to precipitation, and maximum `|storage_change|` was 0.01 mm.
- `route_monthly` on a 2,097,152-node graph with 2048 levels: 0.27 s to compute levels and 0.28 s for annual routing, with zero residual.

## Limitations

- Everything runs on monthly means: no daily temperature distribution, rain-on-snow, sublimation, frozen ground or infiltration-excess storm runoff. Under uniform rain, a cell with monthly rain below Oudin PE in every month yields zero runoff, even when annual rain is substantial (for example, 1500 mm/year spread evenly at the equator). Orogen half-years, or an explicit `wet_half`, give the seasonal concentration that produces surplus.
- The soil, groundwater, snow-cap and snow-partition parameters are uncalibrated global defaults. There is no aquifer geology, glacier dynamics or human use.
- `route_monthly` is flow-through. Lake storage, monthly lake levels and losses that depend on the water balance are not modelled here. Callers supply potential losses.
- Hydraulic geometry and Manning `n` are global priors. Navigation ignores ice cover, tides, locks and canals, and only sees the obstacles it is given. Vessel thresholds are worldbuilding heuristics.

## Delivery adaptations

The service uses the same 365-day synthetic year as the global core. `route_monthly(..., conserve_annual_loss=True)` redistributes an annual loss among months with available water, retaining the annual target instead of silently losing dry-month evaporation. This is an availability-constrained seasonal proxy, not lake storage. Default flow-through behavior remains available. `navigability` accepts either (N,) or (N, 12) outlet flags, allowing a regional section to inherit a seasonally usable downstream route. The service evaluates local section limits before propagating access; it does not copy a parent river's ship class onto all its tributaries.
