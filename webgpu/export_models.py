"""Export the pinned 30 m checkpoint and fail on numerical or graph errors.

Run from any directory with the application's Python environment. Artifacts are
deliberately kept outside the source tree by default.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "terrain-diffusion"))
os.environ.setdefault("HF_HOME", "E:/TerrainDiffusionRuntime/huggingface")
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(ROOT / "terrain-diffusion")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from verify_external_data import verify_pair

from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.onnx.export import _dummy_inputs, export_model
import terrain_diffusion.onnx.export as upstream_export

REVISION = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"
SNAPSHOT = Path(os.environ["HF_HOME"]) / "hub" / "models--xandergos--terrain-diffusion-30m" / "snapshots" / REVISION
DEFAULT_OUTPUT = Path("E:/TerrainDiffusionRuntime/webgpu-models")
NAMES = ("coarse_model", "base_model", "decoder_model")
SIZES = {"coarse_model": 64, "base_model": 64, "decoder_model": 512}
MAX_ABS = 0.003
MAX_RMSE = 0.0003


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, model: EDMUnet2D, name: str) -> dict:
    torch.manual_seed(7700123)
    args = _dummy_inputs(model, batch_size=1, device="cpu", image_size=SIZES[name])
    model.eval()
    with torch.no_grad():
        expected = model(args[0], args[1], list(args[2:])).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    feeds = {spec.name: tensor.numpy() for spec, tensor in zip(session.get_inputs(), args, strict=True)}
    actual = session.run(None, feeds)[0]
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError(f"{name}: ONNX output shape or finiteness mismatch: {actual.shape} vs {expected.shape}")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    metrics = {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
    }
    if metrics["max_abs"] > MAX_ABS or metrics["rmse"] > MAX_RMSE:
        raise AssertionError(f"{name}: forward mismatch {metrics}, limits max_abs={MAX_ABS}, rmse={MAX_RMSE}")
    return metrics


def inventory(path: Path) -> dict:
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(str(path))
    ops = Counter(node.op_type for node in graph.graph.node)
    domains = Counter(node.domain or "ai.onnx" for node in graph.graph.node)
    external = sorted({entry.value for initializer in graph.graph.initializer for entry in initializer.external_data if entry.key == "location"})
    return {
        "ir_version": graph.ir_version,
        "opsets": {entry.domain or "ai.onnx": entry.version for entry in graph.opset_import},
        "operators": dict(sorted(ops.items())),
        "domains": dict(sorted(domains.items())),
        "nodes": len(graph.graph.node),
        "initializers": len(graph.graph.initializer),
        "external_data": external,
        "inputs": [{"name": item.name, "dims": [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]} for item in graph.graph.input],
        "outputs": [{"name": item.name, "dims": [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]} for item in graph.graph.output],
    }


def externalize_base(inline: Path, destination: Path, inline_graph: dict, regenerate: bool) -> dict:
    """Produce the browser-loadable large-model pair from the verified graph."""
    model_path = destination / "base_model_external.onnx"
    data_path = destination / "base_model_external.data"
    if regenerate or not (model_path.exists() and data_path.exists()):
        print("Externalizing verified base initializers", flush=True)
        model = onnx.load(str(inline))
        onnx.save_model(
            model, str(model_path), save_as_external_data=True,
            all_tensors_to_one_file=True, location=data_path.name,
            size_threshold=1024, convert_attribute=False,
        )
        del model
    external_graph = inventory(model_path)
    for key in ("ir_version", "opsets", "operators", "domains", "nodes", "initializers", "inputs", "outputs"):
        if external_graph[key] != inline_graph[key]:
            raise AssertionError(f"External base model changes {key}")
    if data_path.name not in external_graph["external_data"]:
        raise AssertionError("External base ONNX does not reference its weight file")
    initializer_verification = verify_pair(inline, model_path, data_path)
    return {
        "path": model_path.name,
        "bytes": model_path.stat().st_size,
        "sha256": sha256(model_path),
        "external_files": [{"path": data_path.name, "bytes": data_path.stat().st_size, "sha256": sha256(data_path)}],
        "derived_from_sha256": sha256(inline),
        "graph": external_graph,
        "initializer_verification": initializer_verification,
    }


def run(names: list[str], destination: Path, device: str, reuse_existing: bool, resize_up: bool) -> None:
    if not SNAPSHOT.exists():
        raise FileNotFoundError(f"Pinned model snapshot missing: {SNAPSHOT}")
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    report_path = destination / "export-report.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if previous and previous.get("checkpoint_revision") != REVISION:
        raise ValueError("Existing export report belongs to another checkpoint")
    results = previous.get("models", {})
    for name in names:
        began = time.perf_counter()
        print(f"Loading pinned {name}", flush=True)
        model = EDMUnet2D.from_pretrained(str(SNAPSHOT), subfolder=name).eval().to("cpu")
        variant = f"{name}_resize" if resize_up else name
        path = destination / f"{variant}.onnx"
        if reuse_existing:
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Rechecking existing {name}", flush=True)
        else:
            print(f"Exporting {name} at {SIZES[name]}x{SIZES[name]}", flush=True)
            if resize_up:
                original_resample = upstream_export._resample_onnx

                def nearest_resize(x, mode="keep", factor=2):
                    if mode == "up":
                        return F.interpolate(x, scale_factor=float(factor), mode="nearest")
                    return original_resample(x, mode=mode, factor=factor)

                upstream_export._resample_onnx = nearest_resize
            export_model(model, path, device=device, opset=17, image_size=SIZES[name])
            if resize_up:
                upstream_export._resample_onnx = original_resample
        model.to("cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"Inventory and strict CPU comparison for {name}", flush=True)
        graph = inventory(path)
        metrics = verify(path, model, name)
        results[variant] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "graph": graph,
            "forward_comparison": metrics,
            "seconds": time.perf_counter() - began,
        }
        if name == "base_model" and not resize_up:
            results["base_model_external"] = externalize_base(path, destination, graph, regenerate=not reuse_existing)
        report_path.write_text(json.dumps({
            "checkpoint_revision": REVISION,
            "onnx": onnx.__version__,
            "onnxruntime_cpu": ort.__version__,
            "torch": torch.__version__,
            "thresholds": {"max_abs": MAX_ABS, "rmse": MAX_RMSE},
            "models": results,
        }, indent=2), encoding="utf-8")
        print(f"PASS {name}: {metrics}, {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reuse-existing", action="store_true", help="Verify and inventory existing artifacts without exporting")
    parser.add_argument("--resize-up", action="store_true", help="Use nearest Resize for upsampling in place of Tile-based repeat")
    parser.add_argument("--materialize-base-external-only", action="store_true", help="Backfill/check the browser external-data pair from an already verified inline base export")
    args = parser.parse_args()
    if args.materialize_base_external_only:
        report_path = args.output / "export-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["checkpoint_revision"] != REVISION:
            raise ValueError("Existing export report belongs to another checkpoint")
        inline = args.output / "base_model.onnx"
        if sha256(inline) != report["models"]["base_model"]["sha256"]:
            raise AssertionError("Inline base differs from its verified export report")
        report["models"]["base_model_external"] = externalize_base(
            inline, args.output, report["models"]["base_model"]["graph"], regenerate=False,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        run(args.models, args.output, args.device, args.reuse_existing, args.resize_up)
