# Satellite and Magna Urbis close relief

The Python/WebGPU viewer opens **Satellite · Magna Urbis** by default. Its
stable layer ID remains `render-v2`, so explicit links and diagnostics continue
to work. It imports the sibling Magna Urbis Satellite material: mineral ground,
seasonal foliage, snow, ground detail, roughness, light response and forest
population averaging. Both fine tiles and native coarse blocks use it.
Planetary climate and hydrology remain Neural Earth's spatial inputs.

In **Rendering → Terrain amplification**, select **Magna Urbis · Noise + physical
ground** to use the alternative geometry below 30 m. Intensity 1 reproduces the
pinned default Noise parameters. **B** switches before/after; **Shift+B** cycles
available detail modes. The initial choice remains Classic, and each alternative
has its own cache identity. A direct comparison link can use
`?mode=render-v2&amplify_mode=magna&amplify=1` along with its seed/world settings.

This option reconstructs the native DEM, applies Magna Urbis' derivative-weighted
ridged fBm, then adds the ground displacement resolved by the finest geometric
grid (0.46875 m). All noise wavelengths and probes use fixed metres, independent
of camera and tile boundaries. Below LOD -6, the existing bounded C2 interpolation
extends this surface to LOD -12. Our coastal reconstruction is retained;
procedural detail leaves water and terrain signs unchanged. Intensity zero uses
the exact current refinement. Noise is presentation detail, not hydraulic
erosion; flow/deposition guides remain zero and rock/scree guides follow slopes.

The full imported appearance requires WebGPU. CPU/PNG overviews and fallback
images retain the existing seasonal Render v2 approximation. The standalone
Rust/WASM viewer retains its current rendering and native-resolution limit;
it does not advertise the Python geometry modes.

Validation includes execution of the pinned upstream TypeScript as an
independent numerical oracle, overlapping/negative/far-coordinate windows,
ocean/flat terrain, cache and HTTP delivery, UI defaults, and actual WebGPU
rendering in six climate cases plus native coarse terrain. The material harness
writes captures and its measurements under `output/render-v2/`; these are
technical fixture checks, not a generated-world visual acceptance corpus.

See [source and license provenance](../../native/magna_surface/PROVENANCE.md).
