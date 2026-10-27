# Render v2 — snapshot saved on October 23, 2026

The user-requested branch is `codex/render-v2-snapshot-2026-10-23` in both Neural Earth and Magna Urbis. This is the requested state for commit and push, not a claim of finished photorealism. The [French original](render-v2-snapshot-2026-10-23_FR.md) is available alongside this translation.

## Current production state

A single Render v2, its season and diagnostics, shared WGSL/NumPy/GLSL materials, multiscale cover, preservation of beaches and terrain lighting, seasonal snow/water/ice composition, and a regional river mask. The annual snow floor fixes the EF plain that had turned gray in summer. NE production fingerprint: `2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3`.

Validation of this state: 77 Python rendering tests passed; CG typecheck passed. The annual correction has 27 targeted real captures, with unchanged DFC/arid controls. CG evidence and its limitations are in `../city_generator/docs/methods/render-v2.md` from the shared RPG root.

## Work not integrated or not finished

- The clast, stochastic crown and nine-volume experiments were visually rejected. Their tools do not replace the production shader.
- The CC0 example-based ground prototype showed a targeted improvement. Its grass/litter extension, production integration and CG port remain unfinished.
- Actual icebergs were isolated by ablation on marine tiles. Their stretching on the polar map remains to be solved without moving objects between projections. Close crowns remain stylized.
- Coarse cover still needs assessment at the same footprint as fine cover; its large winter shapes come from the forest contribution, not ground grain. The river material remains simple.
- The final 990-image campaign is incomplete. Run `89516…`, with 495 main and 90 hydrology images, was rejected after discovery of the annual snow defect. No verdict carries over to the correction.
- Native coverage reaches 20 of the 24 requested strict combinations; four AF/CFB/DFC/POLAR plateaus remain missing. Additional local searches do not prove they are absent from the whole world.
- The limitations of the 30 m DEM, source hydrology and lake depth filter are described in the implementation document.

Captures, large downloads, caches and generated data remain in ignored directories. Experiment scripts may require these local inputs; their presence on the branch does not mean they are integrated or validated. The already running HTTP server must be restarted to load the latest annual NumPy mirror; the next startup loads the saved code.

See [implementation and limitations](render-v2-implementation.md), [reference protocol](render-v2-reference-protocol.md), and [soil references](render-v2-soil-literature.md).
