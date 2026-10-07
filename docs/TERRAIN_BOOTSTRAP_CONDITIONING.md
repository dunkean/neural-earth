# Native height and physical climate conditioning

`terrain_conditioning.py` is the new five-channel adapter. It imports only `source_distributions`, `latitude` and `lapse_rate` from the existing source-statistics module; no Earth, Macro or A4 height construction is inherited or called. The terrain provider is a lazy import of `terrain_bootstrap.get_heightmap(seed, style)` and owns every height sample, coastline, physical range and raster interpolation.

## Public contract

- `WORLD_PROFILES`: `natural`, `terrestrial-gondwana`, `terrestrial-continents`, `terrestrial-earthlike`, `terrestrial-archipelago`.
- `make_conditioning_factory(seed, world_profile='natural')` caches up to twelve factories.
- `BootstrapConditioning(seed, style='earthlike', heightmap=None, stats=None)` accepts an injected CPU provider for verification.
- The provider implements `sample_height_m(xs_metres, ys_metres)` and returns finite float32-compatible `(len(ys), len(xs))` heights in metres.
- `sample(xs, ys)` returns five physical float32 fields: height metres, BIO1 °C, BIO4 monthly temperature standard deviation ×100, BIO12 millimetres/year, BIO15 precipitation coefficient of variation in percent.
- `sample_raw(cj0, ci0, cj1, ci1)` samples Cartesian coordinates `(index + 0.5) × 7680` metres. For terrestrial factories these fields are already finalized; `finalize` returns a copy.
- `__call__` returns a CPU float32 tensor and applies `sign(height) × sqrt(abs(height))` exactly once. Continuous metre heights must be interpolated by the provider before this encoding. The factory path must not then pass through another import-specific square-root encoder.
- `sample_conditioning_preview` returns `fields`, `elev`, `climate`, and `stage`. `climate[:4]` is the physical BIO1/BIO4/BIO12/BIO15 set and `climate[4]` is lapse in °C/metre. BIO1 is at the supplied terrain height; the server subtracts lapse × positive height once to recover its display baseline.
- `BOOTSTRAP_CONDITIONING_VERSION='native-bootstrap-worldclim-v2'`; `CONDITIONING_SNR=(0.05,0.5,0.5,0.5,0.5)`. SNR selection belongs to the inference configuration; the adapter does not alter pipeline configuration itself.

The domain constant is `(-20e6, -10e6, 20e6, 10e6)` metres. Source latitude uses the existing flat-map convention, with northern latitudes at negative y. The provider's world boundary behavior remains its own responsibility.

## Climate calibration and practical limits

The initial terrestrial climate model uses existing fixed source quantiles by absolute latitude. Four independently seeded coherent fBm fields select ranks within those quantiles. Frequencies are defined in the same global coarse coordinate system, not relative to requested patch dimensions. Noise-to-rank conversion uses the fixed empirical source noise tables; no seed-specific or view-specific min/max, empirical distribution, or quantile is computed.

Version 2 derives each terrestrial climate noise seed by XORing the full unsigned 64-bit world seed with the fixed `0x434c494d41544500 | channel` domain (`channel=1..4`), applying the standard SplitMix64 integer avalanche, then reducing to FastNoiseLite's 31-bit seed range. High world-seed bits therefore influence climate instead of being discarded first: world seeds 0 and `2**31` no longer systematically share climate. Individual 31-bit channel seeds can still collide because that library's seed range is finite; the derivation does not promise an injective map from all 64-bit worlds. Natural's protected seed derivation is unchanged. The version change invalidates terrestrial conditioning caches and records the changed climate arrangement; source statistics and native physical heights are unchanged.

The source temperature table already represents a sea-level baseline. BIO1 adds the upstream precipitation-dependent lapse once:

`beta = clip(-6.5 + 0.0015 × BIO12, -9.8, -4.0) / 1000`

`BIO1 = sea_level_temperature + beta × max(height_m, 0)`

BIO4 uses actual source seasonality quantiles. It is not a temperature-regression residual and does not receive the Natural factory's residual reconstruction. BIO12 and BIO15 retain physical source units. Height passes through unchanged; temperature receives no hard -10/40 °C clip or cold-temperature stretch. Lookup endpoints are fixed source-table endpoints, not a crop-derived clamp. No extra climate clamp is applied by this adapter.

This is a latitude-calibrated climate approximation for comparing the new height provider with separately selected height conditioning strength. It does not claim simulated circulation, mountain rain shadows, continentality or derived runoff. Each channel independently selects a quantile, so the adapter also does not reproduce the full joint distribution of observed climate variables or all physically plausible relationships between them. WorldClim supplies land observations; applying their latitude distributions over ocean is an extrapolation, not an ocean-climate dataset. Those limits can be addressed independently after the baseline's neural behavior is measured. All height style changes use the same climate seed streams, which makes terrain-style comparisons interpretable.

Physical units and fixed source marginal quantiles alone do not establish compatibility with the network's training distribution. Independent channel combinations, their different spatial wavelengths, and the new height geometry can produce combinations absent from training; temperature after the terrain lapse adjustment can also extend beyond observed source temperatures. BIO4 intentionally carries actual WorldClim seasonality rather than Natural's residual-based reconstruction. These source and model distribution limits require neural comparison, including SNR selection, and are not corrected by crop-dependent normalization or hidden field clamps.

[WorldClim's primary documentation](https://www.worldclim.org/data/bioclim.html) defines BIO1, BIO4, BIO12 and BIO15; [WorldClim 2.1](https://www.worldclim.org/data/worldclim21.html) describes the 1970–2000 source dataset, released January 2020. The exact installed raster/statistics conventions are preserved by the shared source-statistics functions.

## Protected Natural behavior

Only the authorized `_noise`, `_natural_stats`, and `NaturalConditioning` reference math was copied. Natural keeps its original independent Perlin channels, fixed quantile mapping, seed-zero repair, temperature clamp/stretch, regression-based seasonality reconstruction and precipitation-seasonality adjustment. Its physical preview sampling and once-only height square root retain the established semantics. Natural is the comparison baseline, not a source for terrestrial height geometry.

The protected preview class groups BIO4 float32 additions differently from the installed upstream factory. Its original output bytes are preserved; the checked upstream comparison agrees exactly in elevation, BIO1, BIO12 and BIO15, and within `rtol=1e-6, atol=1e-4` in BIO4. The observed BIO4 difference on the reference window is at most `6.1035e-5` in the stored std ×100 units. Natural neural inference continues to use the original upstream factory.

## CPU validation

Run `.venv/Scripts/python.exe -m unittest test_terrain_conditioning -v` from the workspace. Thirteen tests passed on 2026-10-07, using injected height providers, fixed small source-statistics fixtures, and the installed upstream factory with its existing statistics file; no Rust build, GPU or neural inference is required.

The checks establish metre-height preservation, channel units, exact protected Natural reference bytes for seed 0/7/large seed, independent upstream agreement with the stated BIO4 rounding tolerance, one lapse application without terrestrial temperature clipping, negative-ocean treatment, distinct terrestrial climate for high-bit world-seed changes on identical constant height, independent climate streams, deterministic overlap/nested windows and request order, row batching and reversed traversal, half-cell sampling, signed-square-root reconstruction, fixed latitude interpolation, preview transport, lazy provider routing, supported cached profiles, empty grids and invalid-provider/statistics handling. No test imports or reads the retired geometry module.

These tests validate the adapter contract. They do not establish the native generator's morphology, real-data climate quality, or the trained network's response; those require the separate CPU geography audit and authorized neural comparison.

## Height conditioning choice

A measured four-style coastal study compared height noise 0.5, 0.2, 0.1 and 0.05 with the four climate values held at 0.5. The 0.05 setting reduced median coarse shoreline displacement in all four cases; three paired native 512-pixel crops also improved without post-NN masks. It is now the terrestrial default. Natural keeps its original checkpoint noise values; switching a configured world back to Natural restores them. Lower noise here means stronger conditioning, not a conventional higher-is-better signal-to-noise ratio. Final climate-v2 tests and raw arrays are reported in TERRESTRIAL_BOOTSTRAP_QA.md.
