# Canonical coarse water profiles

`backend/terrain_hydrology_profiles.py` supplies one water authority for every regional crop of a global reach. It is CPU-only and never imports the server or modifies `HydroWorld`. The service supplies canonical physical Coarse heights and owns caching.

```python
from terrain_hydrology_profiles import build_profiles
from terrain_hydrology_regions import required_profile_reaches

ids = set(visible_reach_ids) | set(required_profile_reaches(world, core_bounds))
profiles = build_profiles(world, ids, sample_height, incision_m=30., conflict_incision_m=60.,
                          fill_m=2., fine_valley_m=15., known=cached_profiles)
water = profiles.level(reach_id, station)
```

`sample_height(x, y)` receives one batched point cloud and returns physical heights in metres. Coordinates are geographic map metres; the caller resolves chart seams. `station` is the **fractional original reach vertex index**, not arc length or a regional vertex index. `profiles.profiles[reach_id]` contains read-only `station` and `water_m` arrays in read-only mappings; the service can copy those records into its entry cache. Empty requests perform no height query.

The package includes the requested downstream closure through `downstream_id`, or `hydraulic_downstream_id` across invisible threshold gaps. Reach samples include every raw vertex and subdivision at most `spacing_m` apart physically. On a sphere the longitude metric is bounded over each segment; cross-pole vertices retain the original station convention. Each station samples a cross-section over the whole regional trunk corridor, 0.75 source cell (14.6 km on the 2048 x 1024 atlas) on either side of the parent line, at most a quarter coarse cell apart. A world without a grid shape keeps half a coarse cell. The 19.5 km source line can lie several kilometres off the Coarse valley that the 240 m trunk actually follows. Version 1 used three samples within ±3.84 km, which saw 112–150 m ridges along seed42 reach 5911 while lower ground lay inside the corridor. Only samples in the parent basin can provide a valley floor; if none are available, the original water reference substitutes and `basin_fallback_samples` reports the conflict.

Reaches are processed from sinks upstream. Let `g` be the sampled corridor valley, `S` the descending source water envelope, `t` the already established downstream level, and `u` the minimum corridor valley of the reach's direct tributaries (lake entries excluded):

- `preferred = reverse_cummax(g - incision_m)`
- `fits = cummin(min(g, u) - fine_valley_m + fill_m)`
- `w = max(min(max(reverse_cummax(g - conflict_incision_m), min(preferred, fits)), S), t)`

Every envelope descends, so `w` descends.
- **No conflict.** If no downstream valley rises more than `incision_m - fine_valley_m + fill_m` (17 m) above any upstream valley, `preferred <= fits` and the version 1 profile is unchanged.
- **Barrier.** Otherwise `preferred` would dam the whole upstream network over a barrier. Instead the water may cut up to `conflict_incision_m` (60 m, the same strong-correction bound the regions and display conditioning use), staying at most `fill_m` above the expected 240 m valley. That valley is the corridor minimum less `fine_valley_m`: on the actual seed42 piece, fine trunk ground lay a median 0–5 m, a 90th percentile 12–15 m, and at most 28 m below it.
- **Junctions.** The tributary term `u` lets a low tributary valley bound the junction it flows into. Without it, seed42 reach 5911 (computed first, downstream) never saw the 5860/5807 lowland behind its ridge. `u` only adds a pure 1-hop upstream dependency, so a profile is still a function of the world alone.

`conflict_lowering_m` reports the lowering relative to the preferred rule. On the seed42 chain, the 5911 head dropped from 109.87 to 79.77 m, with 55.6 m of conflict incision at its ridge. The final point is exactly `t`. At confluences `t` is the downstream profile's first water level; lake entries and exits retain the authoritative lake level. Terminal and mouth endpoints retain their original terminal water level. A fixed lake source below its required downstream endpoint is impossible and raises `ValueError`; it is never silently relaxed into an uphill segment. Other fixed-level/source-cap disagreements are reported as `source_cap_excess_m`.

This preserves the source network's gross height envelope while adapting it to canonical Coarse valleys, before any regional terrain is read. The construction is deterministic for a reach's downstream closure plus the direct tributaries of each closure reach. It is independent of requested siblings, crop order, camera, or regional node ground levels. A shared node has one level: the downstream reach's first water, which upstream reaches take as their target. `known` (reach id -> profile) stops the closure walk at cached profiles and reuses their first water; it never resamples, overwrites or returns them (`reused_known`). The service passes its profile cache so profiles stay immutable across calls.

The 30 m incision parameter is a preference, not a guarantee of physical feasibility: source and fixed-water constraints can require deeper incision. Metadata reports per-reach `max_incision_m`, `max_fill_m`, source-cap excess and basin fallbacks, plus package maxima and sample count. It is a sampled coarse constraint, not a hydraulic solution or a guarantee that 240 m ground and banks fit the water. Regional and display conditioning must report or suppress unresolved physical conflicts. In particular, monotonicity alone does not establish a plausible channel.

Validation: five focused tests cover a ridge that is cut (≤ 60 m) rather than damming the upstream lowland, the unchanged no-conflict profile, `known` reuse, a downstream bump, immutable/original station data, downstream closure and call-order invariance, confluence equality, and fixed lake entry/exit levels. Regional tests exercise neighboring crop portals and unit-link joins with conflicting neural ground.
