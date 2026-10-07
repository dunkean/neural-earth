# B1 correction r2 — source closure and fresh attestation

The cache identity now hashes every `.py` file in the vendored
`terrain_diffusion` package. This includes `models/unet_block.py`,
`common/model_utils.py`, and the package initializer files present in the
checkout. The prior explicit local runtime-source list remains. A test checks
those entries and simulates a changed model-block digest: the resulting world
identity changes without modifying installed files. This closes Opus B1 r2's
P1 dependency gap. The conservative source list still includes server files
outside the reference export path; edits to those files invalidate E1 until
generation identity and build provenance are separated.

The E1 reporter now refuses any CPU or NN artifact whose embedded and metrics
exporter SHA-256 differs from the reporter's current source bytes. The CLI
calls `world.rebuild()` immediately before each site and ablation's checkpoint
export. The report compares conditioned and generated elevation over their
identical 64×64 coarse grid (491.52 km per side), recording metres and
signed-sqrt MAE, signed bias, coast-mask disagreement, and correlation.
Native DEM comparisons remain tied to explicit identical pixel bounds and
requested crop size. The reference NN uses the configured batch-16,
non-canonical profile, so a different crop or export order can still produce
small floating-point differences; this corpus should be compared at the same
crop sizes.

Validation: `python -m unittest test_terrain_macro -q` passed **10/10**.
The fresh command was:

```powershell
.venv/Scripts/python.exe terrain_reference.py --output E:/TerrainDiffusionRuntime/reference-b1r2-final --neural all --neural-site coast_seed0 --neural-site mountain_seed0 --neural-site plain_seed0 --neural-site macro_coast_seed0 --neural-site macro_mountain_seed0 --neural-site macro_plain_seed0 --neural-site coast_negative --neural-site coarse_window_border
```

It produced **14 CPU sites, eight NN sites, 32 neural artifacts** and a
validated `e1_report.json`. Each of the eight reported sites has all four
ablations. The manifest records 107 implementation Python files in this
checkout; its file hashes matched at report generation. The protected A0
coast's matched coarse grid has 1,129.5 m elevation MAE, 0.0815 coast-mask
disagreement, and 0.874 height correlation. These measure conditioned versus
checkpoint-generated regional structure; they do not claim pixel-accurate
coastline reconstruction.

The experimental A3-selected 1024² coast still spans only −11.7 to 21.0 m
over 30.72 km, with 49% land. Its shore remains too flat for visual approval.
High-latitude BIO1 support clipping is also unresolved. `natural` remains the
default; A1–A3 are experimental. Human visual approval and broader NN seed/site
coverage remain open gates.
