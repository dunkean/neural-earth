# Repository Guidelines

## Project Structure & Module Organization

- `backend/`: Flask server, terrain generation, neural inference, scheduling, caching, and CPU rendering.
- `web/`: map/globe viewer, WebGPU renderer, controls, and styles; `web/index.html` is the entry point.
- `tests/python/`, `tests/browser/`, and `tests/fixtures/`: backend regressions, Node-based viewer tests, and reference data.
- `tools/verification/` and `tools/benchmarks/`: live checks and performance measurements.
- `native/`: bundled Orogen, City erosion, and optional Rust bootstrap. `terrain-diffusion/` is an upstream Git submodule.
- `docs/`: installation, architecture, layer guides, and curated media. `webgpu/` contains a separate inference feasibility probe.

## Build, Test, and Development Commands

Run commands from the repository root with the virtual environment activated:

- `git submodule update --init --recursive`: initialize pinned upstream sources.
- `python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`: install the tested dependencies.
- `python launch_terrain.py`: start the local server and open `http://127.0.0.1:8765`; add `--no-open` for unattended startup.
- `python backend/terrain_server.py`: run the server in the foreground.
- `python -m unittest discover -s tests/python`: run Python regressions.
- `node --test tests/browser/*.cjs`: run viewer regressions.
- `node --test webgpu/*.test.mjs`: run probe unit tests.

Follow `docs/installation.md` for Python 3.12, Node.js, CUDA/BF16 hardware, required rasters, and Windows storage defaults. The viewer needs no frontend build step.

## Coding Style & Naming Conventions

Use four-space Python indentation, snake_case functions, and PascalCase classes. Retain `terrain_<feature>.py` and `terrain_<feature>.js` naming. JavaScript commonly uses camelCase functions; match the surrounding indentation and module format. No repository-wide formatter or linter configuration is present; keep edits focused.

## Testing Guidelines

Python tests use `unittest` and `test_<feature>.py`; viewer scripts use Node assertions and `test_<feature>.cjs`. Add regression cases for changed behavior. No coverage threshold is configured. CUDA kernel gates require `TERRAIN_TEST_CUDA=1`; stop the server before GPU verification. For rendering changes, include live captures and relevant verification results.

## Commit & Pull Request Guidelines

History uses concise imperative subjects, such as “Improve biome rendering, map styles and fine-LOD coasts.” Keep commits scoped. PRs should explain changed behavior, list validation and hardware-dependent skips, link relevant issues, and include screenshots for visual changes.

## Configuration & Provenance

Keep model weights, caches, generated outputs, and secrets outside Git. Preserve upstream revision pins, licenses, and provenance notices when changing native adaptations or submodules.
