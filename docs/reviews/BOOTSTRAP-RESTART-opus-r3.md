# Review r3: BOOTSTRAP-RESTART, closure of the r2 findings

**Verdict: ACCEPT WITH LIMITATIONS.** I found no P0 or P1 problems. There are five P2s. Two of them are wording problems that need fixing before the matched-relief result goes into the docs.

You asked me to save the verdict, but you also said read-only and no file edits, so I didn't write anything. The text below is ready to save as `docs/reviews/BOOTSTRAP-RESTART-opus-r3.md`. I only read the code, the tests, the receipts and the four boards. I ran nothing, so I haven't re-checked the "93 CPU tests pass" claim.

## r2 findings now closed (checked against the code)

- **r2 P1-1, stale cache after an identity change: closed.**
  - `valid_physical_report` now compares the stored `world_identity` with the current one before anything else (`terrain_server.py:510-512`). Both the height cache hit (`:737-742`, and again inside the lock at `:750-753`) and the PNG cache hit (`:842-849`) go through it, and the PNG path deletes a stale file before rebuilding it.
  - Each overview mode now has its own identity receipt, `overview.{mode}.json` (`:908-920`, written at `:939`), and the disk-cache record includes it.
  - Browser caching can't serve old data either: tile and overview URLs both carry `world_identity` (`index.html:95,101`).
  - Tests: `test_terrain_navigation.py:226-242` changes the identity, expects a miss and checks the new bytes. `:244-263` would have failed under the old shared receipt.
- **r2 P2-b, errors reaching the client as HTML: closed.** A global handler returns JSON 503 for `RuntimeError`, `OSError` and `SubprocessError` (`:88-92`), so a Git or Cargo `CalledProcessError` from `terrain_bootstrap.py:69,72` is covered. `/api/world` handles it explicitly (`:634-635`).
- **r2 P2-c, snapshot naming a file that isn't running: closed.**
  - `terrestrial_file_snapshot` refuses when the file on disk no longer matches the hash taken at import (`_IMPORTED_SOURCE_SHA256`, `:157-159`). The refusal isn't cached, so every request fails until restart, which is the intended behaviour.
  - The Rust identity is computed lazily, and `_generation_binary` re-checks it (`terrain_bootstrap.py:133-157`), so the identity always describes the binary that actually ran.
  - Test: `:215-224`.
- **r2 P2-a, Natural sharing the conditioning module's identity: accepted as a disclosed limitation.** Natural no longer hashes the Rust bridge, its dependencies or `terrain_bootstrap.py` (`terrain_manifest.py:65-66,78`). It still hashes `terrain_conditioning.py`, as documented in `TERRAIN_BOOTSTRAP_RESTART.md:84-87`. Natural also hashes `terrain_server.py` and other modules anyway, so separating the conditioning file alone wouldn't stop server edits from invalidating its cache. This is consistent with how the identity is designed.

## Matched-relief experiment: what the receipts confirm

- **The comparison is set up correctly.**
  - All four columns use the same native bounds, which `compare_terrestrial_relief.py:136-138` enforces.
  - The noise level that actually ran is recorded (`actual_cond_snr`), and `_build_coarse` reads `cond_snr` when the world is rebuilt (`terrain_inference.py:205`).
  - The receipt is `complete: true` and uses climate v2.
  - Shading uses a fixed palette and a fixed light direction (`terrain_reference.py:120-127,264-269`), so the columns can be compared by eye.
- **The three noise levels sit on 100% land at every site**, so the three masks give identical ratios.
- **Natural is not a usable comparison here.** Its land share on these footprints is 2.8% (Earthlike mountain), 52.8% (plain), 0% (Continents) and 100% (Archipelago). No ratio is computed against Natural, which is correct.
- **The mountain numbers in your summary are confirmed exactly** (noise 0.05 divided by noise 0.5):

| Site | Slope p95 | Relief range (p95−p05) | Fine detail (std after removing ~1 km scale) |
|---|---:|---:|---:|
| Earthlike mountain | 0.632 | 1.109 | 0.687 |
| Archipelago mountain | 0.657 | 0.813 | 0.611 |
| Continents mountain | 1.267 | 1.197 | 1.384 |

## P2 findings

**P2-1: the plain-site figures in your summary don't match the receipt (proven by reading).**
- **Receipt:** `relief-comparison.json:3004-3006` gives slope p95 **0.4066**, relief 0.531 and fine detail 0.485 for the plain. Recomputed from the raw quantiles: 0.0799 / 0.1965 = 0.407.
- **What that means:** the plain loses **59%** of its slope p95 at 0.05. The other slope quantiles lose 58–60% too (p50 0.397, p90 0.402, p99 0.418).
- **Summary:** ".442 slope" doesn't appear anywhere in the receipt, and "none >50% slope loss" only holds for the three mountain sites.
- **Fix:** quote 0.407 and say that the >50% loss happens at the plain.

**P2-2: the "mountain-median" sites are high plateaus, not mountains (misleading label, and missing evidence).**
- **Why:** sites are ranked by parent height only (`selection_score` equals the height, at `:24-26` and `:3034-3036`).
- **What the receipt shows:**
  - Earthlike "mountain" at 0.05 has 54 m of relief over 15.36 km and a slope p95 of 0.033. At 0.5 it has about 49 m and 0.053.
  - The parent is nearly flat: Archipelago parent slope p95 is 0.009 and its fine-detail std is 1.35 m (`:4494-4502`).
  - The Earthlike board looks nearly featureless in all three noise columns.
- **Consequence:** the 0.05 tradeoff has been measured on gentle relief only. The failure the audit worries about (smooth input erasing real mountain ranges) was never sampled.
- **Fix:**
  - Rename these sites to `high-elevation-median`.
  - Add one site per style picked by parent local relief (for example, the top percentile of 3×3 raster range or gradient), plus a site at the parent's height p99.
  - Run the same three noise levels there.
  - This is a gate for describing mountains, not for keeping the default.

**P2-3: the docs don't contain the matched-relief experiment yet.**
- `TERRESTRIAL_BOOTSTRAP_QA.md:136-137` still justifies 0.05 with the coast study alone.
- Neither doc mentions `matched-relief`, the tradeoff or its numbers.
- `TERRAIN_BOOTSTRAP_RESTART.md:94` still says 92 tests.
- **Fix:** add a section with the table above, the corrected plain figures, the Natural land shares and the plateau caveat, plus the decision: "0.05 kept for stronger coast control; no relief gain across all sites; the plain loses about 59% of its slope."

**P2-4: the GPU half of the receipt names comparer bytes that no longer exist.**
- `comparer_sha256` (`8737…`) differs from `cpu_enrichment_sha256` (`1288…`). The comparer that produced the GPU results was edited afterwards, and that version wasn't kept.
- **Fix:** keep a copy of the comparer version that ran, or rerun `compare()` with the current file during the next exclusive GPU slot.

**P2-5: the world identity is rehashed on every cache hit (cost not measured).**
- `world_identity()` re-serialises and re-hashes the whole manifest: the per-file hash list and, for terrestrial worlds, the hypsometry knots.
- One cache hit can do this 2–4 times: in `_tile_coordinates`, in `valid_physical_report`, and in `existing_final_mip` for LOD 1–2.
- **Fix:** keep the identity cached next to `world_manifest` (an `lru_cache` on `(seed, profile)`). This needs a measurement before anyone calls it significant, but fast navigation is the product priority.

## Limitations accepted, not counted as defects
- Polar NN precipitation goes negative and is neither clipped nor masked. Physical climate is not certified, and that is in scope as stated.
- Projection: the archipelago reaches 16.46% on the flat map against a 15% criterion measured on the sphere. This is disclosed.
- Bootstrap heightmap files have no global quota (about 3.8 MiB per world). This was r2 P2-g and is documented.
- Natural's cache is still invalidated by edits to the conditioning module. This is documented.
- The Archipelago Natural column is a land sample from a different world, not a matched reference. It shows Natural can produce fine detail; it isn't a target ratio.

## Gates still open
1. The final run (21 native DEMs, 8 sparse worlds, 32 CPU worlds and Earth512) is being redone with current source hashes. I haven't certified it until the receipt shows `complete` with matching hashes.
2. Byte-for-byte GPU check that the Natural reference tile is unchanged, and that its cache is reused or rebuilt as expected.
3. A real browser LOD session, with measured first-open latency including NN, transport and display. The CPU bootstrap timings (1.18–2.34 s fresh, 37–45 ms from disk) cover only part of that.
4. A high-relief site per style, and corrected docs (P2-1 to P2-3).
5. Visual approval at continental, regional and 30 m scales. I'm not certifying visual quality or performance from these boards. One observation: the Archipelago crop has straight escarpments running along the image axes, at the same place in all three noise columns. I couldn't tie them to the 7,680 m conditioning grid or the 39 km raster cells, so the cause is open.

The claude.ai Google Calendar, Google Drive and Sentry connectors need authorizing in your claude.ai connector settings before they can be used. None of them was needed for this review.