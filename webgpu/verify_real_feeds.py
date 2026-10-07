"""Compare captured production CUDA forwards with native ORT CPU on real feeds."""

from pathlib import Path
import gc
import hashlib
import json
import numpy as np
import onnxruntime as ort

DEST = Path("E:/TerrainDiffusionRuntime/webgpu-models")
CAPTURE = json.loads((DEST / "crop64-coast-manifest.json").read_text())["first_real_forward"]


def array(spec):
    return np.fromfile(DEST / spec["file"], dtype=np.float32).reshape(spec["shape"])


def main():
    report = {}
    for name in ("coarse_model", "base_model", "decoder_model"):
        path = DEST / ("base_model_external.onnx" if name == "base_model" else f"{name}.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        spec = CAPTURE[name]
        feeds = {item.name: array(spec[item.name]) for item in session.get_inputs()}
        expected = array(spec["output"])
        actual = session.run(None, feeds)[0]
        output_path = DEST / f"real-feed-{name}-ort-cpu-output.bin"
        actual.astype(np.float32, copy=False).tofile(output_path)
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        report[name] = {
            "shape": list(actual.shape),
            "max_abs": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "source": "captured unmodified upstream PyTorch CUDA production forward",
            "reference": "ONNX Runtime CPU 1.30.0 on identical captured inputs",
            "cpu_output": {"file": output_path.name, "shape": list(actual.shape),
                           "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()},
        }
        print(name, report[name], flush=True)
        del session, feeds, expected, actual, diff
        gc.collect()
    (DEST / "real-feed-ort-cpu-comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
