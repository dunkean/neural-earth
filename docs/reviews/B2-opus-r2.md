**Verdict: CHANGES REQUIRED.** There is one P1 in the background coarse preparation. It doesn't corrupt terrain, but it breaks resumable preparation and its progress reporting. The fix is small. Once it lands with a CPU regression test, I would rate this block **ACCEPT WITH LIMITATIONS**, without waiting for the optional canonical profile.

I only read code and documents. I didn't run anything, and I didn't look at the outputs of the GPU runs now in progress. I can't certify visual quality or performance: this block has no measurements of either.

## r1 findings: what was fixed

| r1 finding | Status | Evidence |
|---|---|---|
| Opus P1-1: terrain depends on navigation history | **Not fixed in the default; disclosed.** The default still uses latent batch 16. Batch-one latents are now an opt-in profile, so the world's identity is distinct. Acceptable as a residual, but see P2-C on how it is labelled. | `terrain_inference.py:63-77`, `WINDOW_IMPLEMENTATION.md:15-20` |
| Opus P1-2: world-edge tiles never become learned-ready | **Fixed.** At LOD ≥7 the server clamps sample coordinates to the world, the area sampling clamps to the grid, and the preparation plan now adds 9 cells of context around the world. I checked the arithmetic: the farthest climate read stops at row 1312, exactly where the plan stops (1303+9). A test covers tiles straddling the edge at LOD 7 and 12. | `terrain_server.py:348-350,377-381,449-450,293,298`; `terrain_coarse.py:30-32,176-180`; `test_terrain_windows.py:214-218` |
| Opus P1-3: free VRAM changes the world's identity | **Fixed.** The latent batch no longer depends on VRAM, and the profile `name` (which still contains `int(gib)`) is removed from the identity. | `terrain_inference.py:45-78`, `terrain_server.py:53`, test `:94-106` |
| Manifest not tied to the actual world | **Fixed.** `install()` now checks seed, SNR, precision, ablation, profile and batch keys. Only the seed and SNR checks have tests. | `terrain_coarse.py:116-142`, test `:272-289` |
| A full disk quota made requests fail | **Fixed.** `_save_window` now returns `False` and demand continues. | `terrain_coarse.py:321-323`, test `:291-303` |
| Astra R1-01: empty file crashes recovery | **Fixed.** `EOFError` is caught; tests cover an empty file and a truncated header. | `terrain_coarse.py:285,305`, test `:233-240` |
| Astra R1-02: all-zero weights accepted as coverage | **Fixed.** Readiness now hashes the weight channel against the canonical taper, and loading compares it exactly. | `:277-280,302`, test `:253-270` |
| Unbounded `unique` sets | **Fixed.** A 1 MiB Bloom filter per stage; the docs call its counts estimates. | `terrain_window_scheduler.py:27-57` |
| Coarse timing stopped before the GPU finished | **Fixed.** Model timing now synchronises; scheduler stage times are labelled inclusive. | `terrain_coarse.py:211-213,390-392` |
| Background step reading every file | **Mostly fixed**, but a related defect remains: P1-A. | |

## New findings

### P1-A: background preparation never counts windows written by the foreground world (proven by reading the code)

- **Where:**
  - `terrain_coarse.py:364-379` is the only place `step()` adds to `_persisted_indices`, and only for windows it computes itself.
  - `_valid_metadata` (`:240-287`) and `_load_window` (`:289-308`) can remove an index from that set but never add one.
  - The background stops when `complete_windows >= total_windows` (`terrain_background.py:61`), and `complete_windows` is just the size of that set (`terrain_coarse.py:504`).
- **Why it happens in the server:** the foreground world (`terrain_server.py:179`) and the background world (`:211`) for the same seed each install their own `CoarsePreparation` on the same directory. Any window the foreground writes after the background installed:
  1. is missing from the background's set;
  2. gets loaded by `ensure_window` from disk (a full read plus SHA-256);
  3. is loaded a second time at `:379`, which succeeds;
  4. is still never added to the set.
- **Consequences:**
  - Once the truly missing windows are done, every quantum walks the whole plan, doing two payload reads and two hashes (about 229 KB) per foreground-written window, while holding `gpu_lock`. That is my arithmetic, not a measurement.
  - `coverage` never reaches 1, so the task runs until its budget of up to 8192 quanta is used up.
  - A one-off `OSError` during readiness (for example an antivirus lock on Windows) also removes a window from the set for good, with the same effect in a single process.
  - The 2 GiB quota is counted separately by each instance (`_disk_bytes`). With a plan of about 707 MB this rarely binds, but the accounting is wrong.
- **CPU reproduction:**
  ```python
  m, b = _manifest(5), (0, 0, 6*7680, 6*7680)
  fg, bg = _world(), _world(cache_bytes=100_000)
  a = CoarsePreparation(tmp, m, bounds=b).install(fg)
  p = CoarsePreparation(tmp, m, bounds=b).install(bg)
  read_rect(fg, 'coarse', -9, -9, 15, 15)   # foreground persists every planned window
  s = p.step(bg, budget_windows=1)
  assert s['complete_windows'] == s['total_windows']   # fails: 0 < total, network_windows == 0
  ```
- **Fix:**
  - In `step()`, before calling `ensure_window`, check `index in persisted or self._valid_metadata(index)`. If the window is valid, add it to the set and `continue`; that check only reads metadata, not the payload.
  - Also add the index to the set after a successful `_load_window` at `:379`.
  - Better still, keep one shared state object per directory (persisted set, stamp cache, `_disk_bytes`) so the foreground and background instances agree on progress and quota.
  - Add the test above, plus a step where `_valid_metadata` fails once and then succeeds.

### P2-B: readiness checks scale badly on large tiles (cost not measured)

- **Where:** `ready_for_samples` (`terrain_coarse.py:448-465`) calls `complete()` for every pair of y-group and x-group. Each call lists its windows again and runs `_valid_metadata`, which costs two `stat()` calls even when cached. `_coarse_area_ready` (`terrain_server.py:302-306`) adds roughly 100 chunks of about 121 windows each.
- **Rough estimate:** a tile at LOD 11–12 triggers on the order of 10⁴ `stat` calls per check. `learned_tile_ready` repeats this for cached tiles (`:454-457`).
- **Also:** `install()` checks all 6,160 planned files (`:183-184`). The background world is created inside `gpu_lock` (`terrain_server.py:203-211`).
- **Fix:**
  - Collect the set of required window indices once per tile, then check each window once.
  - Use the shared persisted set (from P1-A) as the fast path, and keep `stat`/hash checks for actual loads.
  - Record how long a readiness check takes at LOD 7, 9 and 12, and how long `install()` takes on a full world.

### P2-C: the 0.21 m drift is labelled "upstream", but that was never measured

- **Where:** `WINDOW_IMPLEMENTATION.md:76-79` says "The upstream pipeline itself differs…".
  - The "off" run in `verify_terrain_windows.py:39-43` only removes the scheduler wrappers. It still uses the project's own `cuda-resident-window-v2` runtime: cached normalised weights, GPU windows, and CUDA graphs for base batches ≤4 with eager above that.
  - So the measured drift belongs to the project's pre-B2 runtime profile, not to upstream code.
  - It is one seed, one crop, one split order, measured on v2 code. It is an observation, not a bound.
- **The gate is weaker than it looks:** `request_shape_within_tolerance` (`:98-100`) passes at ≤1 m with no coastline check. The audit itself (§13.3) warns that 1 m is not enough on flat coasts.
- **Fix:**
  - Reword it as "pre-B2 runtime profile, observed 0.21 m on one crop; not a bound".
  - State the consequences: tiles cached at different times in `physical-v1` can disagree at their shared edges, and a revisit after eviction may differ from a cached neighbour.
  - Add `coast_disagreements` to that comparison, as the canonical verifier already does.

### P2-D: the canonical verifier can't tell where a difference comes from

- **Where:** `verify_canonical_latents.py` compares only `world.get()`. That output also includes reconstruction steps computed on the requested crop:
  - `_compute_elev` pads by 6 low-res cells, then applies a Laplacian denoise with reflect-padded `gaussian_blur` and resizes (`world_pipeline.py:1382-1405`);
  - `_compute_climate` works on a coarse array whose size depends on the crop (`:1418-1462`);
  - GPU convolution algorithms can also change with tensor shape.
- **Risk:** the gate requires exactly 0 m (`:92-93`). It may fail, or a past drift may have been misattributed, for reasons unrelated to latent batching. The CPU tests use only `read_rect('decoder')` and never exercise `get()`.
- **The coastline check can pass trivially:** `coast_disagreements == 0` (`:94-96`) proves nothing if a test crop contains no coastline. Nothing checks that any of the four crops does.
- **Fix:**
  - Also record whole-vs-split differences of `read_rect('latent')` and `read_rect('decoder')`. If the windows match exactly but `get()` doesn't, reconstruction needs canonical global blocks.
  - Require at least one case whose land fraction is between 5 % and 95 %.

### P2-E: verifier gaps

- `subcrops_vs_full_after` is computed but not gated (`verify_terrain_windows.py:88`). It is a pure cache replay, so it should be exactly 0.
- `verify_coarse_persistence.py:46-48` should also require `replay_prep.disk_hits > 0`. This is safe today, because `rebuild()` creates a new store (`world_pipeline.py:738-739`), but the assertion would catch a future change.

### P2-F: the cache identity is too broad (costly, not incorrect)

`world_hash` includes the hashes of `terrain_server.py` and `terrain_app.py` and the driver string (`terrain_manifest.py:59-64`, `terrain_server.py:54-57`). Any server edit or driver update throws away the learned coarse (about 707 MB) and the physical caches. Opus raised this in r1 and it is still open. Either document it in `WINDOW_IMPLEMENTATION.md` or derive the coarse cache's identity from only what affects coarse output.

## Residual drift and how honestly it is stated

The default profile's dependence on navigation history is disclosed accurately in substance (`WINDOW_IMPLEMENTATION.md:15-20,76-85`). The docs also say the existing reports predate v3. Not promoting the canonical profile is acceptable for this block. What needs correcting is the attribution and framing in P2-C: "pre-B2 runtime profile, observed on one crop, not bounded, coastline displacement not measured."

## Outstanding gates

1. P1-A fixed, with a CPU test covering two instances and a transient invalidation.
2. The v3 GPU reports (windows and persistence) pass with source hashes matching the reviewed files. The canonical report is informational only and doesn't block this block.
3. The default-profile drift reported with coastline disagreements, and the "upstream" wording corrected.
4. Readiness and `install()` latency measured at LOD 7/9/12 on a fully prepared world.
5. Still unproven, carried to B3/B5:
   - peak GPU memory for the largest production read (`MAX_READ_ELEMENTS` bounds output size, not the dependency working set);
   - E3 navigation p95 times;
   - E4 time and storage for full-world preparation;
   - visual approval of the learned coarse.