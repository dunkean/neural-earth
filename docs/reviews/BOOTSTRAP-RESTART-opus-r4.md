# Review: BOOTSTRAP-RESTART final integration closure

**Verdict: ACCEPT WITH LIMITATIONS.** The admission fix is correct, and I found no P0 and no code-level P1. One P1 remains, and it's about evidence and a decision rather than code: the relief-stress GPU run has finished, but the docs still describe it as pending. It shows that elevation noise 0.05 measurably smooths relief.

This was read-only. I ran no tests or GPU jobs and edited nothing, so the 95-test pass is not independently re-run. I only counted 95 `def test_` across 12 files, which matches the claim.

## What I verified

- **Admission fix** (`terrain_coarse.py:167-176`):
  - Natural maps to `A0`; terrestrial profiles keep their full name. This matches `terrain_manifest.py:100` and `terrain_server.py:133`.
  - The profile checks run before any disk write (`:193-210`), so a rejected admission leaves nothing on disk.
  - The other guards are all still there: seed `:158`, SNR `:160`, precision `:162-166`, execution profile `:177-181`, batch size one `:182`, rebinding `:184-187`, manifest collision `:194-197`, geometry contract `:206-208`.
- **Test** (`test_terrain_server_world.py`):
  - It runs the real server functions (extracted from `terrain_server.py`) with the real WorldPipeline, `configure_world`, manifest builder and CoarsePreparation. Network forwards and CUDA access are forbidden.
  - The asserted error messages are specific enough. If the profile check were removed, install would raise "already has a different persistent identity" instead, so the test would fail.
- **browser-cold.json**:
  - First draws at LOD 4/3/2/1/0 at 4619 / 7045 / 10896 / 12516 / 12682 ms; full coverage at 12925 ms.
  - 10/10 tiles at LOD0, decoder source, 30 m.
  - 22 cache misses, no errors.
  - 80 coarse forwards (4 windows × 20 steps), with noise `[0.05, 0.5, 0.5, 0.5, 0.5]`.
- **browser-warm.json**: 65 / 188 / 250 / 374 / 485 / 552 ms, 22 cache hits, and zero new forwards (counters 36/80/40 before and after).
- **Natural after the fix**: `natural-byte-reference-post-admission.json` is byte-exact. Its source hashes (`terrain_coarse.py` 206d553a…, `terrain_server.py` deba8895…, `terrain_manifest.py` ad0be7f1…) match the manifest recorded in browser-cold.
- **Other post-admission GPU receipts are complete** (post-admission-final, Earth512, relief-stress), each with a `cpu-attestation.json`.
  - Earth512 reproduces the documented values exactly: land 27.65 % → 27.21 %, negative precipitation 16.80 %, one negative BIO15 point.
  - The 0.05 world identities in relief-stress equal those in post-admission-final.
- **Original comparer**: `matched-relief/relief-comparison.json` records `comparer_sha256` 8737…, and `comparer-gpu-source.py` is present. I could not hash it read-only.

## Findings

**P1 (evidence/decision): relief-stress results exist but are not documented, and they show 0.05 smoothing relief**
- Source: `post-admission-relief-stress/relief-comparison.json`, own-land ratios of 0.05 to 0.5:

  | Site | Relief p95−p05 | Slope p95 | 1 km high-pass std |
  |---|---:|---:|---:|
  | Gondwana high local relief | 1.06 | 0.826 | 0.829 |
  | Continents high local relief | 0.614 | 0.843 | 0.753 |
  | Earthlike high local relief | 1.11 | 1.03 | 0.830 |
  | Archipelago high local relief | 0.757 | 0.716 | 0.415 |
  | Gondwana p99 height | 0.687 | 0.708 | 0.658 |
  | Continents p99 height | 0.843 | 0.910 | 0.997 |
  | Earthlike p99 height | 0.975 | 1.055 | 1.055 |
  | Archipelago p99 height | 0.760 | 0.788 | 0.716 |

- So slope p95 drops by 9–29 % on 6 of 8 sites, and high-pass detail falls to as low as 0.415.
- The NN still adds relief over its parent: Gondwana goes from 1123 m to 2085 m at 0.05. The loss is relative to 0.5, not a collapse.
- `TERRESTRIAL_BOOTSTRAP_QA.md:240-243` still says these outputs "restent à exécuter".
- **Fix:** add the table to the QA doc (the 0.1 variant is in the same receipt) and record an explicit decision: keep 0.05 for coastline retention with this measured relief cost, or re-open the choice. No code change unless the decision changes.

**P2-1: server integrity failures are returned as HTTP 400**
- `terrain_server.py:820-821` and `:864-865` map every `ValueError` to 400. That includes admission mismatches, manifest collisions and geometry-contract changes raised from `get_world` inside the job. `browser-before-coarse-fix.json` shows the server defect arriving as 400s.
- The docs say generation endpoint errors become JSON 503.
- **Fix:** raise a dedicated `AdmissionError(RuntimeError)` from `CoarsePreparation.install` (or wrap it in `_create_world`) so the existing 503 handler applies. Keep `ValueError` for bad requests.

**P2-2: the precision and ablation guards are untested, and fp32 can never be admitted**
- WorldPipeline sets `_dtype=None` for fp32 (`world_pipeline.py:369-370`). The mapping at `terrain_coarse.py:163` therefore never produces `'fp32'`, so an fp32 diagnostic world always fails. That fails safe, but the mapping is dead.
- No test covers "precision does not match" or "ablation does not match". The wrong-profile test builds an inconsistent manifest (ablation kept, `world_profile` changed). A real `world_manifest(42,'terrestrial-continents')` installed on an Earthlike world is rejected earlier by the ablation branch, which is never exercised.
- **Fix:** map `None` to `'fp32'`. Add tests for a `_dtype=torch.float16` world against a bf16 manifest, and for a genuine Continents manifest installed on an Earthlike world.

**P2-3: guards are skipped when an attribute is missing (latent, not reachable today)**
- Every check in `terrain_coarse.py:158-181` is wrapped in `hasattr`. A terrestrial manifest installed on a world without `_terrain_world_profile` would cache upstream Perlin conditioning (`world_pipeline.py:685-691`) under the terrestrial namespace.
- The only path that skips `configure_world` is the `ImportError` fallback at `terrain_server.py:187-192`. It can't run, because startup hashes `terrain_inference.py` (`terrain_manifest.py:65`) and would raise `FileNotFoundError` first.
- **Fix:** require `_terrain_world_profile` whenever `manifest['world_profile'] != 'natural'`, or delete the dead fallback.

**P2-4: the Natural "post-admission" byte check bypasses admission**
- `verify_natural_runtime_reference.py` uses `load_pipeline` + `configure_world` + `server.sample_physical`. It never calls `get_world` or `CoarsePreparation.install`, so it doesn't test the persisted-window reload path. Natural admission is covered by the CPU test only.
- That receipt and the harness neural manifests use `terrain_reference.neural_manifest`: precision is written `"torch.bfloat16"` and the hashes (e.g. dc819cf…) differ from the server's (a60a5368… for Earthlike seed 42). QA evidence can't be linked to the served world by hash.
- **Fix:** run the byte check through `server.get_world(42,'natural')`, once cold and once after clearing the RAM tile store so it replays from disk. Also record `server.world_manifest(seed,profile)['world_hash']` in harness receipts.

**P2-5: the warm measurement is a same-process revisit**
- The warm run's `serverBefore` shows forwards 36/80/40 already done and `cached_seeds: ["42"]`: models loaded and the world resident.
- So 552 ms is a hot server with physical-tile disk hits, not a revisit after a restart. Reword the "Revisite, tuiles persistées" column in `TERRAIN_BOOTSTRAP_RESTART.md`. A post-restart revisit has not been measured.

**P2-6: the injected heightmap is not checked against the manifest (hardening)**
- The manifest takes `bootstrap.height_sha256` from `bootstrap_metadata` (`terrain_manifest.py:108`). The conditioning factory loads its heightmap separately (`terrain_conditioning.py:144-147`), and the three caches have different sizes (4 / 12 / 32).
- They stay consistent today only because regeneration is deterministic (the Rayon 1 vs 4 test).
- **Fix:** at install time, assert `world.synthetic_map_factory.heightmap.metadata['height_sha256'] == manifest['bootstrap']['height_sha256']`. It's a string comparison.

**P2-7: docs lag the completed receipts**
- `TERRAIN_BOOTSTRAP_RESTART.md:132-135` cites only the pre-fix Natural receipt.
- `TERRESTRIAL_BOOTSTRAP_QA.md:288-290` cites current-final and current-earthlike-global512 rather than the post-admission-* receipts.

## Accepted as stated

- The identity check costs about 1.25–1.30 ms × 2–4 per cached request; deferring that optimization is reasonable.
- Learned climate is uncertified; the negative precipitation is reproduced post-fix.
- The remaining limits are scoped honestly: draw-submission times rather than GPU presentation, Earthlike seed 42 at one site, a single run per mode, and no sustained cold throughput.

## Outstanding gates

1. Document the relief-stress results and record the 0.05 decision (P1).
2. Make admission failures return 503, and test the precision and ablation branches.
3. Re-run the Natural byte check through the server's `get_world` path, including a disk replay.
4. Still unmeasured:
   - a revisit after a server restart;
   - more than one site, style or run (for percentiles);
   - sustained panning with cold tiles;
   - GPU presentation time.
5. Visual approval of high-relief morphology by you; none is certified here.

Separately: the claude.ai Google Calendar, Google Drive and Sentry connectors need authorization in claude.ai connector settings before they can be used. None of them was needed for this review.