# Durable Earth source statistics

`terrain_world.source_distributions()` retains its single-entry process cache
and also stores exact statistics in
`E:/TerrainDiffusionRuntime/source-statistics-cache/<identity>.npz`.
The disk cache avoids raster decoding and quantile sorting after a process
restart. It does not cache generated conditioning fields or learned terrain.

Identity includes the cache schema, SHA-256 of the statistics computation and
lapse-rate implementation, the ordered five-raster channel mapping, NumPy,
Rasterio and GDAL versions, and content SHA-256 of
all five source rasters plus `synthetic_map_stats.json`. A disk hit still reads
and hashes source files; file timestamps alone never establish freshness.
Climate units, latitude selection, lapse calculations, and quantile arithmetic
are unchanged. Float32 and float64 arrays and the sea-probability scalar retain
their original representations.

Archives load with `allow_pickle=False`. The reader checks the metadata SHA-256,
identity, each array's SHA-256, dtype, shape, and finite values. Missing or invalid
archives trigger recomputation. Writers use unique temporary files in the same
directory, flush and sync them, and atomically replace the final archive. An
unwritable cache falls back to normal generation. Other identities are retained.
Schema 2 includes raster order and GDAL version; schema 1 files are retained but
cannot satisfy a schema 2 lookup.

Deployment also changes the broader world identity: `terrain_manifest.py`
hashes both `terrain_world.py` and `terrain_macro.py`. Editing these files
invalidates persisted coarse and physical terrain cache identities, including
the `natural` profile, even when conditioning values remain byte-identical.
The earlier 6,160-window preparation took 439.6 s; that preparation must be
repeated for the new identity. Deploy these cache edits together with A4 v2
and recalculate once after both files are frozen. The source statistics cache
saves about 0.35 s per cold process in the measured run; it does not remove
this separate terrain re-preparation cost.

Set `TERRAIN_SOURCE_STATS_CACHE_DISABLE=1` to bypass persistent reads and writes.
The process cache still applies: use a fresh process or call
`source_distributions.cache_clear()` when measuring multiple cases.

CPU verification on 2026-10-07 used the current real sources (source digest
`f6a3c368d63cde65`) and compared disabled recomputation, cache population, and a
warm disk load after clearing the process cache. Every array matched byte for
byte with identical shape and dtype; all scalars and source metadata matched.
One local run measured 0.415 s for recomputation, 0.496 s for population, and
0.061 s for the warm disk load. These measurements cover this function only,
exclude interpreter/import cost, and do not establish browser overview or GPU
latency. In particular, the roughly 0.35 s saving does not explain an earlier
11.5 s first browser overview.

`test_source_statistics_cache.py` uses small real CPU GeoTIFF fixtures to check
exact warm-cache round trips without raster reads or quantile sorting, corrupt
archive repair, array integrity validation, source and JSON dependency
invalidation, ordered-raster and GDAL-version identity, computation-source
invalidation, unwritable-directory and write-stage fallback with temporary-file
cleanup, and the disable switch. Permission failures are mocked for consistent
Windows behavior instead of relying on POSIX directory modes.
