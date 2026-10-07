# Bootstrap restart review — Astra r7: UI closure

Date: 2026-10-07. Read-only receipt/source verification; no GPU work, test rerun or production edit.

**No new blocker. The full-world preparation opt-in correction is verified.** Comparing current `index.html` with the preserved prior HTML shows exactly one changed line: automatic preparation now requires `prepare=1`, instead of running unless `prepare=0`. The explicit preparation button remains available. NN code and Python source are unchanged.

Independently verified the prior HTML SHA256 `bb02e9a4abd162f9ca615e662cc9777123994ea2771edc919ff47b14914be8b9` and current SHA256 `afffd309a0e9e317b801fcc327608fb389f48c2ba37cbd91e88656b76175583a`.

`manual-world-preparation-policy.json` passes on current HTML, and its preserved harness matches the recorded hash. This actual-Chrome **mocked-API** regression reports zero preparation POSTs on default startup and random-seed change, the correct seed/profile/8192-window request from the explicit button, and automatic preparation with `prepare=1`. This proves UI policy behavior, not real inference performance.

The separate real-server `browser-ui-final-cache.json` also passes on current HTML: no errors, **22 physical cache hits**, native-view completion in **594 ms**, and 10/10 visible tiles ready exclusively at LOD0/30 m decoder source. All harness source hashes match local files. The raw HTTP manifest has a valid canonical hash and every implementation digest matches current source.

The earlier cold/warm/restart receipts retain their exact prior HTML provenance; their NN/server evidence is not relabeled as a new-HTML run. This final cached run verifies the current UI with the unchanged server. Previous quality and timing limitations remain as documented in r5–r6. No further implementation change is requested.
