# B2 Astra review — revision 3

Date: 2026-10-07. Reviewer: Astra, high. Decision: **ACCEPT WITH LIMITATIONS** for the corrected persistence/scheduler implementation. This does not complete the product's exact request-shape or cold-navigation gates.

Reviewed the shared-namespace correction in `terrain_coarse.py`, its two-instance regressions, `WINDOW_IMPLEMENTATION.md`, relevant server serialization, and the previous Astra findings. Re-ran `test_terrain_windows` and `test_terrain_climate`: **17/17 passed**. No GPU workload was run and no implementation files were changed.

The reviewed `terrain_coarse.py` SHA-256 is `d9ab0029543af45df6c61ad963fe3632aad2035b6e40e8a6c48f5b3613b7614d`. Scheduler and inference hashes remain `dbb420902247815ec097c5464e798aa7621a59c427eccaf5d644629549106daa` and `298f210ca6675630a79e343dcea236f0d3220aa575b0ac033d209b165e541847`. Subsequent implementation changes need their own validation.

## Revision-2 defect resolved

Foreground and background preparations installed before either produces a window now adopt one live ledger keyed by the resolved persistence namespace. It shares completed indices, verification stamps, payload bytes and quota exhaustion. Existing manifest and tensor-contract checks precede adoption, and a conflicting in-process budget is refused. The weak registry does not retain abandoned worlds by itself.

- **Peer completion:** a foreground write immediately updates the background completion set. A completed background `step()` returns without replaying every persisted window. The new regression checks full peer coverage and that the background model is never called.
- **Shared quota:** both writers admit a new payload against the same byte count, and exhaustion propagates to both. The two-instance test fills a 1,152-byte fixture budget through the foreground, then attempts another background window and verifies that actual payload bytes remain 1,152. Demand's existing computed-RAM fallback remains intact.
- **Peer invalidation:** a failed metadata or demand check removes shared membership and verification state. A known invalidation reconciles actual payload bytes and can release the shared exhaustion flag. The new test truncates a persisted window, discovers it through the peer, and verifies that the original preparation also loses completion and reports the corrected bytes.
- **Readiness cost:** duplicate contributors within one sampled readiness request are checked once. Normal peer adoption does not introduce a full directory scan on every background quantum; scanning occurs at first namespace installation or known-file invalidation.

The server still serializes neural generation through its GPU lock, including `prepare_coarse_quantum()`. This acceptance covers that single-process serialized writer model. The ledger is not a cross-process locking or transaction system, as the documentation now states. The configured quota counts window payloads per world; it does not establish an aggregate all-world disk bound or include every sidecar/temp-file byte. These are disclosed limits, not a recurrence of the proven two-instance accounting defect.

No further proven P1/P2 defect was found in this correction. The earlier v3 shape/identity/taper checks, full digest/finiteness checks on demand, corruption recovery, RAM fallback, numerical-profile separation and bounded instrumentation remain covered by the existing review and CPU regressions. Cheap readiness is still distinct from verifying the entire payload digest, and external file damage is discovered on access rather than by a continuous scrub.

## Evidence limits and final gates

The existing real-checkpoint persistence and scheduler reports validate their recorded source bytes. They do **not** certify this shared-ledger correction's final source hash. `WINDOW_IMPLEMENTATION.md` now states that explicitly. Run and retain the final source-bound GPU replay after the implementation snapshot is frozen; preserve exact weighted-channel replay, zero replay coarse-model calls, and zero added scheduler drift as the relevant gates. Missing final replay evidence is a limitation here, not a newly proven arithmetic defect.

The default batch-16 runtime's measured whole/subcrop discrepancy of about 0.210693 m remains an upstream/local-profile limitation. Optional batch-one canonical latents remain **NO-GO**: the four-seed evidence reports 0.096–0.324 m whole/subcrop discrepancies and roughly 2–11.9 m differences from the accepted baseline. Equal maximum discrepancies across request orders do not prove pairwise array equality, and the discrepancy's cause has not been isolated. The revised documentation no longer promotes that experiment as an exactness fix.

Full-world preparation time, peak dependency memory, real-model corruption/cancellation/resume behavior, broad default-profile edge/revisit fidelity and B3 cold-navigation throughput remain separate evidence needs. There is no basis here for an ultra-fast cold 30 m claim, complete-world preparation claim, or promotion of a new numerical profile.

**Disposition:** revision-2's proven shared-cache P2 is closed. Accept the corrected conservative implementation with the explicit scope above, pending the planned clean GPU evidence run for final release certification.
