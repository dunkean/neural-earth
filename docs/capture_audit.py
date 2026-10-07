"""Capture the local evidence used by the design, without running inference.

The session observations below are historical samples, not a controlled benchmark.
Run again only to intentionally replace the evidence snapshot.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"
SNAPSHOT = Path(r"E:\TerrainDiffusionRuntime\huggingface\hub\models--xandergos--terrain-diffusion-30m\snapshots") / REVISION


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


weights = []
for name in ("coarse_model", "base_model", "decoder_model"):
    path = SNAPSHOT / name / "diffusion_pytorch_model.safetensors"
    weights.append({"stage": name, "bytes": path.stat().st_size,
                    "sha256": checksum(path),
                    "config": json.loads((SNAPSHOT / name / "config.json").read_text())})

data = []
for name in ("etopo_10m.tif", "wc2.1_10m_bio_1.tif", "wc2.1_10m_bio_4.tif",
             "wc2.1_10m_bio_12.tif", "wc2.1_10m_bio_15.tif", "synthetic_map_stats.json"):
    path = ROOT / "terrain-diffusion" / "data" / "global" / name
    data.append({"file": name, "bytes": path.stat().st_size, "sha256": checksum(path)})

samples = [32, 40, 42, 39, 38, 55, 27, 29, 29, 87, 48, 64, 53, 35, 44]
audit = {
    "date": "2026-10-07",
    "purpose": "Read-only snapshot supporting TERRAIN_REALTIME_DESIGN.md",
    "classification": "session observations, not a controlled benchmark",
    "upstream_commit": "e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230",
    "model": "xandergos/terrain-diffusion-30m",
    "model_revision": REVISION,
    "pipeline_config": json.loads((SNAPSHOT / "config.json").read_text()),
    "weights": weights,
    "data": data,
    "app_hashes": {name: checksum(ROOT / name) for name in
                   ("terrain_server.py", "terrain_app.py", "index.html", "requirements-lock.txt")},
    "recorded_export": json.loads((ROOT / "generated" / "terrain.json").read_text()),
    "natural_tile_log_examples": [line for line in (ROOT / "server.log").read_text().splitlines()
                                  if line.startswith("Tile 42/8/") or line.startswith("Tile 42/9/")],
    "gpu_session_observation": {
        "utilization_percent_samples": samples,
        "mean_percent": sum(samples) / len(samples),
        "memory_used_mib_approx_range": [5216, 5288],
        "memory_total_mib": 24576,
        "duration_seconds_approx": 8.7,
        "limitations": ["Global device utilization, desktop included",
                        "Historical samples from the ongoing navigation session",
                        "No cache controls or per-kernel attribution",
                        "Not a Tensor Core occupancy measurement"]
    },
    "proposed_slos_are_measured": False,
    "future_engine_was_benchmarked": False
}
path = ROOT / "docs" / "audit-session.json"
path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Saved {path.name}: 3 weight hashes, {len(data)} data hashes, historical observations")
