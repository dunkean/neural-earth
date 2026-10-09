# Developing Neural Earth

## Repository layout

| Location | Content |
| --- | --- |
| `terrain_*.py` | Server, physical generation, inference, persistence and CPU rendering |
| `index.html`, `terrain_*.js`, `terrain_styles.json` | Browser viewer and WebGPU rendering |
| `native/` | Vendored adapters, licenses and provenance |
| `terrain-diffusion/` | Pinned upstream model submodule |
| `research_projects/` | Pinned comparison/reference submodules |
| `test_*`, `verify_*`, `benchmark_*` | Tests and reproducible verification tools |
| `tests/fixtures/` | Reference corpus consumed by validation code |
| `docs/` | English public guides |
| `docs/design/` | Prospective design notes; French is permitted |
| `docs/images/` | Curated public screenshots and capture provenance |
| `output/`, `tmp/`, `generated/` | Ignored outputs, scratch work and runtime artifacts |

Keep upstream submodules and pretrained weights unchanged. Adapt behavior in this project's wrappers. Preserve numerical/physical CPU and GPU contracts. Source-generation changes should produce new cache identities; appearance-only changes should reuse neural heights.

Public prose, interface labels, tooltips, accessibility text and user-visible errors are in English. Preserve stable API identifiers, URL parameters and checkpoint names. Audits, code review reports, experiment logs and implementation work documents belong in an external archive, not public docs. Generate evidence under ignored `output/`; promote only selected screenshots to `docs/images/`.

## Validation

Python tests are plain unittest modules; JavaScript tests are standalone Node scripts:

```powershell
.\.venv\Scripts\python.exe -m unittest test_terrain_identity test_terrain_render test_terrain_pedology test_terrain_orogen_stages
node test_terrain_lod.cjs
node test_terrain_map_tools.cjs
node --test webgpu/conditioning.test.mjs
```

Run focused suites relevant to the edit. GPU suites must run with the server stopped. Browser verification uses Playwright (`PLAYWRIGHT_PATH`) and Chrome (`CHROME_PATH`); current defaults point to the reference machine's installation. Live verification scripts need the localhost server; other scripts create their own fixture server.

```powershell
node verify_terrain_render_live.cjs
```

Record actual source resolution and loaded LOD when capturing neural terrain. A finer displayed grid is not evidence of finer learned detail. Avoid mixing historical receipts with current behavior. Prospective design notes must identify unimplemented proposals.
