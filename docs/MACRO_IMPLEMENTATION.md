> Rapport historique : les profils Earth et Macro A1-A4 sont retirés depuis la reprise du 7 octobre 2026. Voir [le nouveau bootstrap](TERRAIN_BOOTSTRAP_RESTART.md).

# Reference and experimental macro geography

`natural` remains the protected checkpoint baseline. The new CPU composer in
[`terrain_macro.py`](../terrain_macro.py) is an experiment for organizing global
landmasses, ridges, basins, and climate before the learned coarse stage. Its
output is five physical fields at a 7,680 m coarse cell: signed elevation in
metres, BIO1 in °C, BIO4 in monthly temperature standard deviation ×100,
BIO12 in mm/year, and BIO15 in percent. `__call__` applies the signed square
root to elevation once; checkpoint means and standard deviations remain in
`WorldPipeline`. All four variants retain `cond_snr=[0.5]*5`. No macro field
is used as a hard output mask.

| Variant | Elevation | Climate | Purpose |
|---|---|---|---|
| A0 | Installed Perlin and statistics | Installed Perlin and finalize | Protected natural reference |
| A1 | Seeded finite continental supports | A0 raw climate noise finalized against A1 elevation | Isolate geographic organization with natural lapse and BIO4 coupling |
| A2 | A0 physical elevation | Latitude and regional WorldClim quantiles | Isolate climate |
| A3 | A1 plus finite ridge and basin objects | A2 | Combined experiment |

A0 reads the installed `synthetic_map_stats.json` and reproduces upstream
conditioning on square windows to within 0.001 in each transformed input channel in the test. Seed
zero uses the upstream intended wrapped noise seeds 1–5; upstream's `seed or
random.randint` otherwise makes zero nondeterministic. A0's arbitrary-point
preview samples the same noise functions at real world coordinates and marks
the result as *conditioning*. It does not present an uncomputed NN result.

The v2 macro objects are created once per unsigned 64-bit seed. Continental
supports are clustered ellipses with coordinate warp and near-shore texture.
The sea threshold and elevation rank tables are measured on one fixed global
lattice, never on a request rectangle. Each ocean and land rank maps back into
the natural checkpoint's empirical elevation quantile table. A3 adds long
uplift ridges and regional depressions. A2/A3 interpolate source WorldClim
quantiles by absolute latitude, then apply the source lapse-rate convention to
BIO1. The climate fields are bounded by A0's support on the fixed calibration
lattice for that seed. Before clipping, 20.6% of seed-zero A3 BIO1 cells fall
outside that support; the polar climate is still experimental. These are
statistical and geometric approximations, not tectonics,
hydrology, glacier modelling, or a trained control pathway. The visible
macro shapes still require the A0–A3 NN comparison and user visual approval.

The canonical world identity is built by
[`terrain_manifest.py`](../terrain_manifest.py). `build_manifest(seed,
ablation='A0', checkpoint_source=..., inference_profile=...)` embeds the
checkpoint revision and config, its weight hashes, ETOPO/WorldClim and
statistics hashes, operative source hashes, 64-bit seed as a decimal string,
coordinate convention, noise, model windows, backend, precision, and numerical
profile. `world_identity(manifest)` rejects incomplete or tampered manifests.
Absolute installation paths are excluded from the hashed payload. A code,
model, source, or numerical profile change produces a different cache identity.
The server and scheduler must use that `world_hash` when persisting results;
metadata alone does not protect a cache if its key ignores the hash.

The fixed sites live in
[`reference_corpus.json`](reference_corpus.json). They include both A0-selected
and A3-selected coasts, mountains, and plains, plus seed zero, negative
coordinates, and a coarse window boundary. Run the CPU export with:

```powershell
.venv/Scripts/python.exe terrain_reference.py --output E:/TerrainDiffusionRuntime/reference
```

Every site and variant exports `conditioning.npz` (five physical fields,
five model inputs, land mask, bounds), elevation PNG with one fixed palette,
central height profile CSV, and channel metrics. Each site's conditioning board
places A0–A3 together. The index lists manifest hashes and explicitly says
whether neural results were captured. `--quick` skips file hashes for previews
and creates an incomplete manifest that cannot name a persistent world.
`--site coast_seed0` limits the export.

For an already loaded and correctly configured `WorldPipeline`, call
`export_neural_intermediates(world, site, output)`. It exports the actual
weighted coarse stage, weighted latent stage, low-frequency latent, final
30 m DEM, climate and a fixed-palette DEM PNG in a bounded native crop. The
CLI `--neural all` path invokes the installed CUDA model
for A0–A3, uses the same fixed native focus at every profile, and creates
stage boards and `e1_report.json`. The default crop is 256², with three
selected 1024² regional crops. It should be run only when the
GPU is available. The server's macro profile integration must configure the
matching factory before binding.

Validation in `test_terrain_macro.py` covers A0 upstream agreement, 64-bit
seed zero, negative coordinates, crop order and shared borders, ablation
isolation, field units, broad global distributions, latitude, palette, and
manifest refusal of incomplete identities. A CPU conditioning comparison is
diagnostic only. Acceptance requires the same sites at learned coarse, latent,
and 30 m decoder stages, with protected coast and mountain forms reviewed
side by side. The reference corpus is a deliberately small first gate; more
stratified sites and observed good/bad user locations should be added before
replacing natural.

The first E1 export is at `E:/TerrainDiffusionRuntime/reference-l0-l4`:
14 CPU conditioning sites, six sites with A0–A3 checkpoint coarse/latent/30 m
stages, complete manifests and fixed-palette boards. The protected A0 coast
crop is 50% land and spans −202 to 356 m; A0 mountain spans 4,967 to 6,230 m.
At A3-selected locations the candidate makes a coast and mountain, but the
coastal 128² crop spans only −1 to 7 m and looks overly flat. Its mountain
spans 6,115 to 6,878 m after reducing ridge uplift to stay within the
conditioning source support. The candidate changes the terrain at A0's old
coordinates, as a new geography should; those same-coordinate differences
are measurements, not an automatic failure. The flat candidate coast and the
small corpus mean A3 has **not** passed the visual gate. Keep `natural` as the
default while experimenting with margin variation and regional relief bands.

That first export is historical B1 r1 evidence. Its neural manifests omitted
the actual graph and batch settings, and its stage summaries spanned different
areas without stating that fact. The revised exporter writes CPU conditioning
and NN results under separate `worlds/<world_hash>/<site>/` paths. It builds
the NN manifest from the configured world and installed checkpoint, records
the actual inference version and numerical settings, and binds each NPZ to
the manifest hash, seed, ablation, exporter hash, bounds and file SHA-256.
`e1_report.json` rejects stale or mixed products and source changes. Full
stage summaries are labeled as different-footprint context. Separate metrics
interpolate coarse, latent and conditioning elevations onto the same DEM pixel
centers. Fixed-light hillshade, unsaturated fixed colors, center profiles,
and coarse/latent boards support the visual review. The latest attested run is
at `E:/TerrainDiffusionRuntime/reference-b1r2-final`: 14 CPU sites, eight NN
sites across seeds zero and 42, 32 neural artifacts, and a validated E1
report. The world identity now includes every Python source in the vendored
`terrain_diffusion` package, including model blocks and utilities. E1 also
checks each artifact against the current exporter and compares conditioning
and generated elevation on their common 64² coarse grid. Every NN site starts
from a rebuilt world. This conservative identity includes some server sources
which are not called by the reference exporter, so unrelated edits can
invalidate its attestation. A3's selected 1024² coastal crop spans −12 to 21 m across 30.72 km,
which remains a visual quality concern. `natural` stays the default.
