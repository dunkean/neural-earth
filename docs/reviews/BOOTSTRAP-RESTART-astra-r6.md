# Bootstrap restart review — Astra r6: evidence closure

Date: 2026-10-07. Read-only supplement to r5. No additional GPU work, tests or production changes by this reviewer.

**The evidence pending in r5 is now complete. No new proven blocker was found.** The height/navigation default at altitude noise 0.05 remains accepted with the already published coast, fine-relief, rectilinear-feature and learned-climate limitations. This is not certification of a complete learned planet or universal latency.

The three `final-hardened*` reports are now `complete=true`: 21 ordinary native exports, 24 stress variants and nine global probe sets in total. Each associated CPU attestation passes, reports byte-exact numerical arrays against its prior run, and matches the SHA256 of the exact final report read here. These close the final-identity refresh gap; the existing quality conclusions are unchanged.

`natural-server-replay-final.json` uses actual `terrain_server.get_world(42, 'natural')` and BF16 server sampling on two separately constructed worlds. Both 391,444-byte elevation/climate payloads equal the original saved reference exactly. The recreated world reuses four persisted coarse windows with **zero coarse forwards**, while recomputing 16 base and 25 decoder invocations. This distinguishes real coarse persistence from merely retaining the first world's RAM output. All implementation hashes in its admitted manifest match current local source.

Independently checked the final Chrome cold, warm and post-server-restart receipts. All three pass with no errors, finish with 10/10 visible tiles ready at LOD0/30 m decoder source, and have valid canonical manifests reconstructed from raw HTTP JSON. Every harness source hash and every manifest implementation hash matches the frozen local source.

| Actual Chrome session | Native coverage complete | Physical cache | Neural forwards during session |
| --- | ---: | --- | --- |
| Cold | 12.443 s | 22 misses | 80 coarse, 36 base, 40 decoder |
| Warm | 0.570 s | 22 hits | Cached physical payload path |
| After server restart | 0.569 s | 22 hits | Zero coarse/base/decoder |

The restart run begins and ends with no resident cached world seeds and zero model forwards, demonstrating persisted physical tile reuse. These timings retain the harness's separate native-view timing origin and measurement conditions; they are not universal first-open or frame-rate guarantees.

The final guards remain those reviewed and tested in r5 (`terrain_coarse.py` SHA256 prefix `5e8b3ded`, `terrain_server.py` prefix `f7d93ba3`). The parent's full 98-test pass is additional evidence, not an independently rerun test claim here. No further implementation change is requested by this review.
