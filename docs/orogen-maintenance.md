# Orogen upstream maintenance

## Active integration

As of 2026-11-12, Orogen v3 replaces the previous bundled generator directly.
The pinned upstream commit is
[`ed466f9db71292e912d8d0160e689a67a994f2d9`](https://github.com/raguilar011095/planet_heightmap_generation/commit/ed466f9db71292e912d8d0160e689a67a994f2d9),
v3.0.1, released 2026-11-09. The earlier 2026-10-10 review of `cc2662b4`
was a tracking baseline, not the origin of the old snapshot.

The adopted components are Generate's present-day geological world, the common
finish (erosion, hydrology, ice, geology and food data), and the new monthly
climate. No Evolution timeline UI or upstream rendering shaders were integrated.
No saved worlds required compatibility, so V1 snapshots and GPU transformations
have no migration path. Full source/licensing details and local changes are in
[PROVENANCE.md](../native/orogen/PROVENANCE.md).

## Contracts to preserve

- Physical kilometres/metres without quartic conversion, clipping or coastal uplift.
- Coordinate conversion: upstream z north to application y north, consistent longitude.
- Twelve monthly physical fields and WorldClim BIO1/BIO4/BIO12/BIO15 units.
- Independent relief, finish and climate commands; finish identities also include climate.
- Configured physical inputs and the external City erosion bridge.
- Immutable stage identity, integrity checks, cancellation and CUDA atlas projection.
- Same generation inputs for the server and browser/WASM data paths.
- Existing application rendering; shader work is outside this migration.

`upstream/js/planet-worker.js` only changes the Delaunator import and adds the
`retainedTerrain` adapter export. `present/belts.js` also records the actual
back-arc footprint for existing resource inputs, without changing the algorithm.
The other bundled upstream modules are copied unchanged. Future updates should review that dependency closure,
physical units, climate defaults and stage inputs, then update the revision and
source notices explicitly.

The coastline calibration from V1 is unnecessary with direct physical heights.
Upstream still has an open target to reduce overly mountainous coasts; that is
a morphology issue distinct from the former near-zero coastal conversion slope.

## Validation

From the repository root, with the Python environment active:

```shell
python -m unittest discover -s tests/python -p "test_terrain_orogen*.py"
python -m unittest discover -s tests/python -p "test_terrain_generation.py"
python -m unittest discover -s tests/python -p "test_terrain_conditioning.py"
python -m unittest discover -s tests/python -p "test_terrain_koppen.py"
npm run test:wasm
node --test tests/browser/test_terrain_startup_preferences.cjs
```

The atlas checks require CUDA. City erosion additionally requires the optional
wgpu runtime; CUDA spatial search requires CuPy/NVRTC. Do not run GPU checks
beside the terrain server. Use CHROME_PATH for a system browser if needed.
