# Render v2 real-world reference corpus

[French original](render-v2-reference-protocol_FR.md).

The planned corpus contains **60 distinct centers × five zooms × three views = 900 PNGs**. Images come exclusively from the viewer on actually generated tiles of world `7985403534069300490`. Orogen atlas statistics select centers; they are not evidence of native terrain or final appearance.

The manifest and results reside in `output/render-v2-reference/`, separate from the first captures in `output/render-v2-40km/`. Centers common to initial lists are deduplicated to one millimeter and retain original IDs in `aliases`. The corpus combines the initial matrix, other real situations, exact user positions, replacement candidates and three coastal/arid additions. Two further centers explicitly target a beach already visible in a real capture and an ocean point in pack ice; their native samples and provenance are recorded.

| Square extent | Forced LOD | Required source | 1024² image footprint |
| --- | --- | --- | --- |
| 153.6 km | 3 | real latent, 240 m | 150 m/pixel |
| 40 km | 0 | real decoder, 30 m | 39.0625 m/pixel |
| 10 km | 0 | real decoder, 30 m | 9.765625 m/pixel |
| 2.56 km | 0 | real decoder, 30 m | 2.5 m/pixel |
| 160 m | 0 | same decoder, 30 m | 0.15625 m/pixel |

The last zoom checks enlarged material and crown composition on the 30 m DEM. It **does not demonstrate submeter terrain** and triggers no negative-LOD refinement. Views are Render v2 July, reference Render3D and Render v2 January. July is southern winter and January southern summer below the equator. The Render3D comparator retains its own historical contract; its label alone does not give it the v2 calendar.

Extents are squares in the DEM's geographic chart, not geodesic squares of equal physical dimensions at every latitude. The source atlas has 19.53 km spacing in this chart; the 33 × 33 bilinear preselection samples are not 1,089 independent observations.

## Measured families and terrain

Six families are explicitly grouped: AF = Af/Am; CFB = Cfb/Cfc; BSK = BSk/BSh; ARID = BWh/BWk; DFC = Dfc/Dsc; POLAR = ET/EF. A family name does not mean every situation belongs to a pure Köppen class. Fractions of all classes are retained.

| Requested terrain | Measurement criterion on the actual 40 km square |
| --- | --- |
| Plain | Height standard deviation < 45 m, P90−P10 < 130 m, slope P95 < 0.15 |
| Plateau | At least 19% high flat surfaces (height ≥ P60 and slope < 0.06), more than 8% slopes > 0.2 and P90−P10 > 300 m; visual plateau confirmation required |
| Mountains | Standard deviation > 160 m, P90−P10 > 450 m and slope P95 > 0.30 |
| Rolling | Standard deviation ≥ 40 m and P90−P10 ≥ 120 m after excluding previous categories |

These thresholds are selection criteria, not a universal geomorphological taxonomy. They are evaluated in plain, plateau, mountains, rolling order and are not changed to turn a failure into success. A wooded basin surrounded by a rim can satisfy some plateau numbers while failing visual review. Coastal and landless situations are recorded separately.

## Preparation and execution

```powershell
.venv/Scripts/python.exe tools/verification/select_render_v2_reference.py
node tools/verification/review_render_v2_reference.cjs --plan
```

These commands start no terrain browser, server request or inference. The first tool reads only the atlas, lists and local receipts. `--measure file.json` measures already present native tiles without requesting generation. Oceans with no land pixels produce `null` land fields and empty biome fractions.

After explicitly freezing production code and the manifest:

The freeze still requires visual preflight of all six families at latent and decoder levels, plus a real EF plain, the beach and marine pack ice. Marine pack ice does not replace the permanent terrestrial snow check. Technical success (exact sources, files and counters) does not mean rendering acceptance. A major defect absent from Render3D stops the campaign: partial captures remain in their original version, without renaming or use to validate the next fix. Preflight `db083bf3841a…` was stopped after a white vermiculated network appeared on DFC at LOD3; its 60 main and 30 hydrology images describe that rejected state, not a completed final gallery. The next preflight `790e34695a49…` retains 105 main and 75 hydrology images. Close inspection revealed translucent crowns on the beach; hydrology comparison also showed disagreement between the fast physical trace and the displayed regional trace. These remain evidence of defects to fix, not a final gallery merely because the protocol succeeded. Every new campaign uses its own freeze.

Pass `89516e9f2c58…` was stopped after 495 complete main images and 90 hydrology images. A real wet EF plain whose warm month remains below zero became gray-beige in July while Render3D retained its white cover. V2 used only instantaneous snow and omitted the permanent annual component. This defect blocks acceptance; evidence and verdicts remain in that rejected run. No image or judgment transfers to the corrected version.

```powershell
node tools/verification/review_render_v2_reference.cjs --freeze
node tools/verification/review_render_v2_reference.cjs --run
```

`--run` uses the existing server, loads the shared recipe and lets the viewer request required tiles. It may therefore populate caches for new centers. It installs or starts no other model or server. `REVIEW_CASES=id1,id2` limits an invocation for incremental resumption. A completed image is reused only if its file and SHA256 match exactly the same code and manifest version. Each version has its own capture subdirectory; old evidence remains intact.

The campaign may split remaining centers between at most two invocations in separate working directories. Each retains the same complete manifest and production freeze; only `REVIEW_CASES` selects its batch. Merging copies PNGs without transformation, verifies SHA256 and retains original receipts in `worker-reports/`. It creates no terrain or images and transfers no visual judgment between versions. Stop the shared-directory runner before merging to avoid concurrent writers. The hydrology supplement also counts toward the limit of two simultaneous capture browsers.

## Cache warmup before the final campaign

```powershell
node tools/verification/warm_render_v2_reference.cjs --plan
$env:REVIEW_CASES='af-rolling,cfb-rolling'
node tools/verification/warm_render_v2_reference.cjs --run
```

Without `REVIEW_CASES`, the driver visits all 60 centers. It uses the existing viewer for one 40 km working view per center at forced LOD0 and real 30 m source. Hydrology, peripheral prefetch and below-LOD0 refinement are disabled; sea remains enabled to compute oceanic centers. It checks world identity, cache profile and source stages before starting. Resumption also verifies native arrays and receipts still exist on disk for that identity.

`output/render-v2-reference/warm/` is separate from the final campaign. These working PNGs prepare caches, without comparing dates or counting toward the 900 images. Warmup can compute new neural tiles; coordinate its launch with other GPU work. `--plan` makes no request and opens no browser. Rendering may still evolve during warmup: only the post-freeze campaign can demonstrate consistent quality for one version.

## Contracts and reading results

The current campaign uses identity `bd36737badf0cedf99200ba94bf7dc761070aca686a98784373247c86efe87dd`, cache profile `96d1b3a5fb584090`, with metadata in `output/render-v2-reference/world-info-bd36737b.json`. The old identity `f5cddc89751589636d1052bfa46529f6f0a400b1fd1b176a35dbb14ce3590a1a` and receipts remain historical. The previous manifest is archived; no physical receipt is renamed or migrated. Source, terrain, erosion and climate IDs are unchanged, but that does not demonstrate neural pixel identity. Every center must therefore be remeasured in the new native caches. The selector's `--world-info path.json` explicitly sets this working version and propagates its identity to the runner.

- Hashes of all viewer JavaScript and CPU rendering modules are frozen. A code or manifest change stops the pass; versions are not mixed.
- The capture and measurement protocol also has a frozen hash; changing tools during the pass is rejected. PNG dimensions are checked in the file header as well as on the canvas.
- World identity and generation stages are checked by a metadata GET. Each image's cache keys must identify that world.
- Each capture requires `pending=0`, a ready renderer, expected actual LOD/source/resolution, exact camera, 1024² canvas, expected mode and date. A conditioning preview or old fine DEM does not satisfy the 240 m latent contract.
- `Render sea` is enabled so purely oceanic cases are actually computed at the requested LOD. Prefetch, grid, HUD and minimap are hidden/disabled for this session. Hydrology is disabled; only DEM sea and coast are included in this corpus.
- Native statistics cover the actually decoded 40 km square. Receipts must state `decoder`, 30 m and the expected identity; partial coverage is rejected. Biomes are reconstructed from the source's continuous climate on DEM heights every 120 m. This classification is a measurement tool, not a field observation.
- Morphological thresholds, biome fractions and mixtures are retained separately from protocol acceptance. A requested plateau measured as mountain remains a selection failure. Visual quality cannot be inferred from green percentage or technical success.

## Gallery

```powershell
node tools/verification/audit_render_v2_reference.cjs
```

The audit produces `coverage.json` and `coverage.md`: one row per combination of four terrain types and six families. A cell is measured only if a real DEM confirms requested terrain and family fraction. The strict 80% and representative 70% thresholds remain distinct; a natural transition is not artificially made a pure biome. A source candidate never counts as a measurement. `reliefAccepted` and `status: accepted` judgments in `reviews.json` are required to mark a cell visually accepted. No automatic reassignment of a mountain to a requested plateau fills the matrix.

Two indicators are retained. `requestedTargetMatches` measures preselection target success. `actualNativeCoverage` measures terrain and families actually present in the DEM independently of the initial target. Thus a center requested as plateau but measured as mountain with 99% CFB remains a plateau-preselection failure and can serve as a CFB mountain reference without renaming it. Identity, native fractions, failed target and provenance are retained. Visual acceptance requires `actualReliefAccepted: true` and `status: accepted` in the review linked to the current `runId`; measurements alone do not validate rendering appearance. A report from an older world identity is ignored for current counts.

Warmup of all 60 centers under identity `bd36737b…` was fully measured: 18 strict targets out of 24, 20 including representative transitions; coverage of actually measured terrain reaches 20 strict cells out of 24. The 60-situation portfolio is therefore not a complete four-terrain × six-family matrix. Four gaps remain: AF, CFB, DFC and POLAR plateaus. Among 21 candidates, three gaps were closed at native level: AF plain, BSK plateau and POLAR plain. Thresholds were not changed. These selection findings remain separate from the final 900-image campaign and visual validation.

An additional bounded search without generation requests measured 820 further 40 km squares fully covered by current native caches: 147 AF, 151 CFB, 206 DFC and 316 POLAR. The 1,148 offsets without complete coverage were rejected. None meets all three plateau criteria; the best AF reaches 17.34% high flat surfaces, below the 19% threshold. The four gaps remain open. Code, exact 30 m measurements and provenance are retained in `output/render-v2-reference/cached-plateau-search/`. This is not proof of absence of plateaus throughout the world, only an explicit search of already available terrain.

Reviews are bound to the version by `runId`; an old judgment is not reused for a new render version. Example structure:

```json
{"runId": "id-copied-from-report", "sites": {"af-rolling": {"status": "accepted", "reliefAccepted": true, "notes": "Comparison at five zooms and both dates."}}}
```

Counts of accepted strict cells and accepted representative transitions are kept separate in `coverage.json`. A visually valid transition at 74% of its family can remain useful without being announced as a pure 80% cell.

For the two targeted centers, the runner also measures the native 160 m square: sea presence for `user-pack-ocean`, mixed shore for `warm-coast-beach-detail`. This geometric check demonstrates neither sea-ice quality nor visual beach preservation; those still need PNG review.

```powershell
node tools/verification/assemble_render_v2_reference.cjs
node tools/verification/assemble_render_v2_reference.cjs --sheets --require-complete
```

The index and per-center pages show five zooms and link to full-resolution PNGs. Sheets group at most four centers and five zooms per view. Assembly reads only actual PNG captures; it creates no synthetic DEM or texture. Without `--require-complete`, status remains explicitly provisional and missing cells are shown as such. Human judgments can be recorded by ID in `reviews.json`; their state remains distinct from the 900 technically verified captures.

Each captured cell also offers “Open in Neural Earth”: shared world, center, extent, size, forced LOD, layer, season and hydrology month are taken from the receipt. This opens the current local viewer; the PNG and its hash remain the frozen-version reference if software changes later.

## Shared land and hydrology entry point

The assembler accepts `REVIEW_REFERENCE_BASE` for the main set or hydrology supplement. Expected site, zoom, style and image counts come from each manifest: 60/900 for the main set and 6/90 for hydrology.

```powershell
$env:REVIEW_REFERENCE_BASE='output/render-v2-reference'
node tools/verification/assemble_render_v2_reference.cjs
$env:REVIEW_REFERENCE_BASE='output/render-v2-hydrology-reference/captures-reference'
node tools/verification/assemble_render_v2_reference.cjs
```

These commands prepare HTML pages without a browser or viewer request. The common entry point is `output/render-v2-reference/index.html`: 66 situations, five zooms and 990 planned images. Gallery links are relative and work when opening this local file. An unassembled set has an explicit waiting page. Indexes report each set's `runId`, world identity and production hash; the combined campaign is complete only when both sets are complete and share production and world. `gallery-summary.json` retains this information without merging receipts or capture versions.

Hydrology pages summarize evidence per image: effective month, lake/river sample counts, maximum cut and actually painted overlay. The historical vector control is identified as such when it has no physical raster. Native 40 km measurements remain required, including for all six hydrology sites. `--require-complete` applies to the selected set, rejects missing files or measurements and checks frozen manifest identity. PNG SHA values are checked when present in receipts. Complete assembly of two different production versions is rejected; no index presents it as a coherent 990-image reference.

The regional-mask protocol also expects `geometryCurrent`: network revision must match the latest terrain arrivals after the refresh delay. For V2 fast, it requires `renderer.riverDisplay.ready`, `regional-mask` mode, the same `frameKey` as the overlay acknowledgment and the requested month. A vector fallback remains useful in the product but does not validate this GPU path. The runner actually reads the submitted RGBA16float texture and records wet and solid pixels and integrated coverage. A valid empty mask measures 1 × 1 with zero wet pixels. Older physical river-class samples remain recorded as sources; they no longer serve as evidence of the rendered trace.

`--sheets` adds PNG sheets using a CPU browser on a local assembly server. It never opens the terrain viewer on 8765. Assembler sources are distinct from the capture protocol: navigation or layout improvements do not invalidate already captured images.
