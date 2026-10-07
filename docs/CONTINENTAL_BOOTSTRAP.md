> Rapport historique : les profils Earth et Macro A1-A4 sont retirés depuis la reprise du 7 octobre 2026. Voir [le nouveau bootstrap](TERRAIN_BOOTSTRAP_RESTART.md).

# A4 continental initialization

**V1 is rejected for default promotion.** The Opus review found a zero-gradient
shore band at coarse-cell scale and hard-saturated conditioning ridges. The
re-centred shoreline images below document the original run; choosing another
focus does not correct those input defects. V2 changes those formulas and
has now replayed the original coordinates through the NN, as recorded below.
After the final Astra high and Opus 5.5 CLI reviews and the real browser gate,
the root promoted A4 as the experimental UI initialization default. The natural
reference remains selectable; this does not certify a complete learned planet.

A4 (`macro-a4`, currently `continental-bootstrap-v2`) supplies structured geography to
the five conditioning inputs of the installed 30 m Terrain Diffusion checkpoint.
It preserves the `natural` reference and historical A1–A3 variants. There is
no trained model change, output land mask, DEM flattening or post-NN terrain.
Coarse conditioning noise remains 0.5 for all five channels.

The old natural frequency 0.05 is appropriate for local variation but gives
thousands of separated land components over the 40,000 × 20,000 km world.
A3 finite elliptical supports reduced fragmentation, but elevation ranked by
continental position made wide flat coasts and concentrated highlands at
continental centres. Independent ridge objects did not reliably reach shores.

A4 creates five seeded curved, branched continental supports. One fixed global
512 × 256 lattice selects a sea offset for 32% land. Noise perturbs the signed
support distance at two scales, making coves and peninsulas while preserving
large ocean basins. Height is independent of continental distance: attached
finite range segments, plateaus, lowlands and broad basin depressions compose
the interior. V2 uses a variable 10.5–145 km shore ramp, adaptively shortened
for low inland heights. The ease-out `t(2−t)` has nonzero slope at sea level:
the sea side reaches about −111 m at signed-distance parameter d=−7,680 m
and −213 m at d=−15,360 m. Land reaches at least roughly +60 m at d=+7,680 m
even for the 65 m lowland floor. These are parameter-distance statements,
not guarantees for every physical grid step: the perturbed distance field
has variable gradient. Actual coarse-cell crossing distributions are measured
separately below. Inland values above 70% of the positive
source bound approach that bound with a C1 exponential shoulder, preserving
crest variation without a hard clipped plateau. Shelves reach −750 m by
100 km, then transition to deep ocean over roughly 1,000 km. Ocean depth
still has a narrow distribution without organized trenches or ocean ridges.
Small near-shore islets can remain; the ocean is not filled by independent
Perlin islands. These are geometric input priors, not simulated tectonics.

The five fields are signed elevation metres, BIO1 °C, BIO4 monthly temperature
standard deviation ×100, BIO12 mm/year and BIO15 %. The factory applies signed
square root to elevation exactly once, leaving checkpoint normalization to
WorldPipeline. A4 reuses A2/A3 latitude-conditioned WorldClim quantiles and the
source precipitation-dependent lapse convention. Elevation is bounded by the
installed source quantile table. Climate is bounded by the natural fixed-lattice
support. These seed-dependent natural-lattice bounds are a heuristic, not
established checkpoint training bounds. Roughly 20.6% of unbounded BIO1
cells fall outside that heuristic support and require clipping. The export
records preclip fractions, bounds and checkpoint z-score diagnostics; it does
not certify polar realism or the checkpoint's undocumented training coverage.
V2 splits clipping by latitude and elevation. For seeds0/42, BIO1 outside
fractions within absolute-latitude bands are 0% at 0–30°, 0.15%/0.43% at
30–60°, and 62.7%/62.4% at 60–90°. Clipping is concentrated near the poles
but is not exclusively polar. Seed-independent climate bounds remain future work.

## Relationship to the Rust generator

The regional Rust/WASM generator is not executed or linked. V1 ported two exact pieces
of algebra from `city_generator/rust/crates/core/src/coast.rs`: the clamped projection onto a finite segment with
interpolated endpoint radii in `Coast::distance`, and the independent inland
height multiplied by a short variable cubic shore ramp in `Coast::apply`.
`ContinentalBootstrap.capsule_distance` reproduces the segment formula; its
rounded ends and varying radii have a focused regression test. V2 retains the
exact capsule distance and independent-inland-height shore architecture,
but replaces the regional cubic easing with a nonzero-gradient planetary
adaptation. V2 does not use the exact Rust land-ramp formula. The ridge
motif uses squared inverted absolute noise as in `Noise::ridged`, but does not
copy Rust's full feedback-octave noise implementation. Python uses the existing
FastNoiseLite Perlin and a different RNG and planetary layout. This is a
documented portable adaptation, not a binary-equivalent Rust port.

Read source snapshots: `coast.rs` SHA-256
`ce823d68b4c40647cf9e3d630b3b545e3c936e85bf74df10fbbed6b277835fad`;
`noise.rs` SHA-256
`772b4b4a4bc9c1358f226feeb42ab4c6af92692d9b3567e0db03a15724bf3b3b`.
No city_generator source was modified. Its regional 1024² erosion/drainage
grid was not expanded to planetary dimensions; none of its positive-only
terrain constructor assumptions are imposed on signed neural DEMs.

## Evidence and use

CPU export:

```powershell
.venv/Scripts/python.exe terrain_bootstrap_preview.py --output E:/TerrainDiffusionRuntime/continental-bootstrap-v2-final --width 1024 --replay-sites E:/TerrainDiffusionRuntime/continental-bootstrap-v1-r2/cpu-report.json
```

The export contains five physical channels, global topology metrics, fixed
palette full-world boards for A0/A3/A4, plus selected coast and ridge boards
at 768 km and 30.72 km widths. Titles explicitly mark conditioning. These
are not NN outputs. At 1024 × 512, seed zero has 4,145 natural land components;
the five largest contain 10.0% of natural land. A4 has 22 components, with
99.91% of land in the five largest and 99.97% of water in one ocean. Seed42
has 3,946 natural versus 12 A4 components, with A4 top-five coverage 99.99%.
Land area is 32.01%/31.99%; the five supports merge into three/two masses of
at least 1M km², respectively, and residual small components are coastal islets.
The historical v1 land below 100 m was 3.1%/3.0% instead of A3's 13.9%/13.8%.
Its seed0 analytic coast input spanned −9 to 1,031 m over 30.72 km; this
analytic 30 m sample was not the actual 7.68 km lattice received by the NN. These statistics
support the inspected CPU boards; they do not replace NN visual validation.

`terrain_bootstrap_validate.py` is the opt-in CUDA gate. It reads the CPU
report, verifies every recorded input/visual source hash (world, macro, stats,
preview/reference/diagnostic helpers and five source rasters), then exports actual coarse, latent,
and 1024² native 30 m DEM stages at the same selected physical coordinates
for A0/A3/A4. Artifacts carry the full configured-world manifest and file
hashes through the existing reference exporter. Each world is rebuilt.
V2 boards distinguish analytic 30 m conditioning, bilinear physical values
from actual 7.68 km conditioning cells including interpolation halo, and NN DEM.
It saves matched conditioning/native boards and reports topology separately
for each footprint. By default it also evaluates a 32 × 16 stratified global
grid through the actual coarse network for A0 and A4 at seed zero, including
all overlapping contributors at each point. The board explicitly labels this
sparse point probe: it is not an exact LOD11 area mean or a complete learned
world. It detects large-scale NN land/ocean changes before a full background
preparation. `--global-width 0` skips the probe. Run only with the root
coordinator's exclusive GPU slot:

```powershell
.venv/Scripts/python.exe terrain_bootstrap_validate.py --cpu-report E:/TerrainDiffusionRuntime/continental-bootstrap-v2-final/cpu-report.json --output E:/TerrainDiffusionRuntime/continental-bootstrap-v2-final-nn --profile A4
```

Focused checks: `python -m unittest test_terrain_macro test_terrain_bootstrap -q`.
They cover the protected A0 and historical samples, capsule algebra, three
seeds, large land/ocean topology, source support, crop invariance, exact input
transform and distinct versioned manifests. Default promotion belongs to the
root integration decision after actual NN and LOD11 visual checks.

## Final V2 conditioning diagnostics, 7 October 2026

`continental-bootstrap-v2-final/cpu-report.json` is the final CPU receipt with
the schema2 exact source-statistics cache, corrected source docstring and all
11 input/diagnostic source hashes. It keeps the original four foci and adds
the median and weakest actual sampled shore-pair foci for each seed, with no
best-relief selection. The inventory covers 14,983/14,461 unique sign-changing
7.68 km pairs in 3×3 neighborhoods around every 1024-wide overview shore edge.
It is a sampled inventory, not the complete worldwide coarse lattice.

The minimum/p05/median/p95/max absolute height steps are
2.34/54.47/182.56/927.44/4,320.14 m at seed0 and
3.65/58.37/189.10/1,037.51/4,064.39 m at seed42. Both endpoints lie within
−20..+20 m for 1.17%/0.98% of pairs. Central differences of the distance field
over ±7.68 km give median gradient norms 1.84/1.90, with sampled minima
0.087/0.098. The weakest height pairs have strong gradient norms 1.79/2.55 but
small crossing-axis components 0.018/0.021: these sample tangential crossings,
not a uniformly flat distance field. The seed0 weakest site is near latitude77°;
the limited and undocumented polar training coverage applies to its NN result.

An independent ETOPO reference uses all source-raster shore edges, the same
flat world coordinate convention and bilinear sampling at 7.68 km centres.
Its 161,937 sampled pairs have median 59.64 m and p95 390.26 m height steps.
The 10 arc-minute source is being upsampled; it is not a native 30 m coast
reference. The observed A4 medians are about three times that sampled reference,
but the inventories are not matched: A4 starts from 39 km overview edges and
examines about ±11.5 km around them, which can miss a crossing up to 19.5 km
away. ETOPO starts from its finer source edges and bilinear upsampling smooths
steps. The factor is an observation of these inventories, not a precise ratio
of planetary coastal distributions. A4 favors steeper shores and does not
reproduce Earth's coastal distribution. Selection labels saying "complete
sampled inventory" mean all deduplicated records within that sampling, never
the whole coarse lattice.

Land below 100 m is 1.89%/1.81%; land above 1,500 m is 19.84%/20.76%.
Soft compression starts at 4,069.85 m and affects 2.18%/2.20% of land.
Raw inland maxima 6,553/6,879 m become 5,394/5,466 m; at p99 the reductions
are 81.56/79.22 m. This preserves varying crests but compresses the tallest
conditioning ranges; zero hard-clipped fraction alone does not certify relief.

## Final V2 NN evidence, 7 October 2026

`E:/TerrainDiffusionRuntime/continental-bootstrap-v2-final-nn/neural-report.json`
contains the final current-source run: the original four coast/ridge foci,
four additional median/weakest sampled shore foci, and the 512-point learned
global probe. All eight native DEMs are finite. The global probe agrees with
the input sign at all 512 points, with 33.40% sampled land. This is sparse
point evidence, not global coast certification or a complete LOD11 area mean.
The global export stage took 54.93 s including initial model loading/warmup;
the eight native export stages took 10.08–11.72 s each. These include receipt
and CPU export work and are not isolated inference benchmarks. The GPU slot
was released after the successful run.

The four original results below remain valid for the final source identity.
`source-only-array-comparison.json` verifies the NPZ/metrics/manifest chains
for the previous and final V2 artifacts and compares raw DEM, climate, both
coarse arrays and three latent arrays. All 28 arrays are byte-identical, with
maximum difference zero. The cache-schema/source-documentation changes
altered manifests without changing these numerical outputs.

The four additional coast boards were inspected at their automatically
selected coordinates, with no re-centering or output modification:

| Site | Actual coarse bilinear land | NN land | NN height range | Water components / main-water share |
| --- | ---: | ---: | ---: | ---: |
| Median seed0 | 38.31% | 28.84% | −561..421 m | 8 / 99.9988% |
| Weakest seed42 | 50.95% | 54.96% | −549..533 m | 7 / 99.9949% |
| Median seed42 | 60.09% | 49.47% | −792..642 m | 2 / 99.9996% |
| Weakest seed0, near 77° latitude | 45.84% | 7.04% | −456..18.5 m | 54 / 94.61% |

The first three show open sea, varied coastal lowlands and learned hills.
The weakest seed0 case is a remaining failure of local conditioning fidelity:
the NN removes most planned land and leaves very low shelf/islet surfaces
with several water components. Its board is
`bootstrap_weakest_coast_seed0-native-board.png`. Climate clipping and unknown
polar training support are possible contributors, not an established causal
diagnosis. This case limits claims of reliable polar or weakest-crossing
coasts even though V2 corrects the source's zero-gradient shore formula.
No subsequent terrain formula or NN output adjustment was made to conceal it.
The final reviews accept these limits for an experimental initializer default.
The real browser gate confirms default A4 and explicit return to natural;
see `docs/RUNTIME_OPTIMIZATION.md`. The polar failure remains part of the evidence.

`continental-bootstrap-v2-final/polar-scope.json` scopes this risk on the saved
sampled pairs without rerunning the NN: 3,438/14,983 (22.95%) at seed0 and
3,170/14,461 (21.92%) at seed42 have absolute midpoint latitude above 60°.
The share with both heights within ±20 m is 0.785%/0.442% within that polar
subset, versus 1.291%/1.134% outside it. These conditioning-pair statistics do
not estimate NN coastline failure frequency or explain the failed polar crop.
The supplement binds the original CPU report and pair NPZ hashes and records
the centre/latitude formulas; no new source or GPU result is implied.

CPU regression tests `test_bootstrap_diagnostics.py` verify x/y pair axes,
deduplicated coordinates and midpoint positions on planar coasts, then both
longitude and latitude raster mappings with a synthetic real GeoTIFF. They
exercise half-pixel convention and north/south orientation; two tests pass.

The initial V2 `historical-comparison.json` binds comparer source SHA `0a7206…`.
The current comparer was extended for the final source-only replay and has a
different SHA. Its earlier full source was not archived, so that exporter
version cannot be reconstructed from this workspace alone. The historical
report/input/metrics/manifest hash chains remain intact and separately labelled;
the final replay has its own current-source receipt. No old source SHA is
represented as matching the later comparer.

## Initial V2 NN evidence before final cache metadata, 7 October 2026

The initial V2 CPU receipt is `E:/TerrainDiffusionRuntime/continental-bootstrap-v2/cpu-report.json`.
It preserves the four original v1 coast/ridge physical foci. The analytic
coast inputs now span −379–2,331 m and −480–4,092 m. The original ridge inputs
vary over 5,083–5,277 m and 5,060–5,220 m; neither is a hard-clipped plateau.
The full-world hard positive clip fraction is zero for both seeds. Seventeen
focused tests pass, including offset-grid topology, exact crop invariance,
nonzero shore gradients at half/one/two coarse cells, retained soft-crest
variation and source-guard refusal for world/stats/preview changes.

The new A4-only CUDA run is `E:/TerrainDiffusionRuntime/continental-bootstrap-v2-nn/neural-report.json`.
It contains one 512-point global coarse probe and four original-focus
1024² native DEM artifacts, each with configured-world manifests and hashes.
Global land signs still match conditioning at all 512 points (33.40% land).
Probe reporting contains land fraction, finite checks and quantiles, with no
planetary area/component claims. The actual conditioning lattice and analytic
30 m input are separate board rows. The global stage took 61.96 s including
first model loading/warmup; four artifact stages took 10.77–12.23 s each.
These export wall times are not isolated inference benchmarks.

At the **original** coast foci, seed0 now has 37.80% water, elevations
−483.7–3,109.7 m and two water components; its largest water body contains
99.9995% of water. Seed42 has 20.02% water (v1: 2.47%), elevations
−68.9–5,035.3 m and 42 water components (v1: 269), with 99.67% of water in
the main sea. Both inspected boards show actual open water, varied shores,
and detailed inland mountain terrain. Seed42 retains a few offshore islands
and small water components; the former polygonal near-sea-level band is not
visible in this crop. The ridge DEMs span 4,256–7,202 m and 4,011–6,181 m,
with preserved learned ridges/valleys. These are four inspected sites, not
a global guarantee of coast quality or absence of every shallow lowland.
Residual shallow values remain: the native |height|<1 m fraction decreases
from v1's 5.56% to 1.19% at seed0 and from 5.20% to 4.06% at seed42. The
review's −0.09..0.82 m band decreases from 1.78% to 0.66% and from 3.99% to
2.02%, respectively. The source-gradient defect is corrected; this does not
prove the NN has eliminated every very shallow shore or island surface.

`terrain_bootstrap_compare.py` builds a CPU-only matched-footprint comparison
from the original A0/A3/A4-v1 artifacts and new A4-v2 artifacts. It verifies
the NPZ→metrics→manifest/internal-field chain and explicitly labels historical
baselines. The comparison receipt and four fixed-palette native boards are
in the v2 NN directory. No old baseline is represented as a new GPU run.
The complete coarse planet and actual LOD11 browser presentation remain
separate root integration gates. No macro mask or height edits follow the NN.

## Historical V1 NN validation, 7 October 2026

The exclusive RTX3090 run completed: 512 actual learned coarse points each for
A0 and A4 at seed zero, then four sites × three profiles with 1024² native
DEM, coarse and latent artifacts. All native DEMs are finite. The A4 global
probe preserves the input land sign at all 512 sampled points (33.40% land);
no sampled open-ocean point became an island. A0 sign agreement is 93.75%.
Do not use connected-component counts from this 32 × 16 sparse probe as actual
planetary topology: independent points can alias and falsely connect masses.
The complete 1024 × 512 CPU topology remains a conditioning measurement.

The initial seed0 coast NN crop spans −11.8 to 1,694.7 m and contains 27.35%
water. Its fixed-palette board shows detailed hills and branching mountain
relief inland, variable shoreline and open water. The initial seed42 crop
contains only 2.47% water: its learned shore moved relative to the selected
input focus. That original result is retained, not replaced or masked. Ridge
DEM ranges are 4,674–7,553 m at seed0 and 4,362–6,910 m at seed42, with detailed
learned ridge/valley geometry in both inspected boards.

A focused second run (`terrain_bootstrap_shore.py`) selects the nearest actual
A4 neural shoreline in each original DEM, records the offset, then samples
A0/A3/A4 at that same new physical focus. These re-centred images and artifacts
are separate from the original cases. The offsets are (+4,320,+120) m at
seed0, distance 4.32 km, and (−2,760,+6,960) m at seed42, distance 7.49 km.
This is approximately one conditioning cell, not a guarantee of exact coastal
alignment. The re-centred A4 crops contain 41.16% and 16.85% water and span
−15.5–1,224.7 m and −97.0–4,613.5 m, respectively. Both inspected boards show
actual sea, varied learned shores and detailed inland relief. The seed42 coast
includes a coastal plain below a high chain. No DEM values were modified.

Reports and boards are in `E:/TerrainDiffusionRuntime/continental-bootstrap-nn`
and `E:/TerrainDiffusionRuntime/continental-bootstrap-shores`; CPU boards remain
in `E:/TerrainDiffusionRuntime/continental-bootstrap-v1-r2`. The separate
executed shore report records the original report hash, selection method and
offset; its per-row original NPZ binding is supplied by the separate
CPU-only `shore-provenance.json` supplement. The supplement verifies original
NPZ/metrics/manifest identities and internal fields, and hashes all six current
shore outputs plus both boards. It preserves the executed report and exporter
hash; the executed exporter source is archived separately. Both original A4
manifests matched all 127 current source/model files when this CPU attestation
was made. This is provenance verification, not a rerun with the subsequently
updated shore exporter. Future exporter rows include the three original
NPZ/metrics/manifest hashes directly. The global stages took
49.59 s for A0 (includes first model loading/warmup) and 50.92 s for A4; local
artifact stages took roughly 6–11 s each including CPU export/receipt work.
These are validation wall times, not isolated NN inference benchmarks or a
profile speed comparison. The complete coarse planet and actual browser LOD11
presentation still belong to the root's background-preparation/browser gate.
