"""Build minimal first-Conv channel/layout probes without touching the GPU."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

OUT = Path("E:/TerrainDiffusionRuntime/webgpu-models")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = onnx.load(str(OUT / "decoder_first_conv_repro.onnx"))
    weight6 = onnx.numpy_helper.to_array(original.graph.initializer[0]).astype(np.float32)
    x6 = np.fromfile(OUT / "decoder_diag_39_reference.bin", dtype=np.float32).reshape(1, 6, 512, 512)
    source_node = original.graph.node[0]
    entries = []
    baseline = None
    for size in (64, 512):
        offset = (512 - size) // 2
        original_input = x6[:, :, offset:offset + size, offset:offset + size].copy()
        for channels in (4, 6, 8, 12):
            if channels < 6:
                x = original_input[:, :channels].copy()
                weight = weight6[:, :channels].copy()
            else:
                extra = channels - 6
                x = np.pad(original_input, ((0, 0), (0, extra), (0, 0), (0, 0)))
                weight = np.pad(weight6, ((0, 0), (0, extra), (0, 0), (0, 0)))
            node = deepcopy(source_node)
            node.input[0], node.input[1], node.output[0] = "x", "weight", "y"
            model = onnx.helper.make_model(onnx.helper.make_graph(
                [node], f"first-conv-cin{channels}-size{size}",
                [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, channels, size, size])],
                [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64, size, size])],
                [onnx.numpy_helper.from_array(weight, "weight")],
            ), opset_imports=list(original.opset_import))
            model.ir_version = original.ir_version
            onnx.checker.check_model(model)
            prefix = f"conv_cin{channels}_{size}"
            model_path = OUT / f"{prefix}.onnx"
            input_path = OUT / f"{prefix}_x.bin"
            reference_path = OUT / f"{prefix}_cpu.bin"
            onnx.save(model, str(model_path))
            x.tofile(input_path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
            y = session.run(None, {"x": x})[0]
            y.tofile(reference_path)
            if channels == 6:
                baseline = y
            max_vs_6 = None if channels < 6 else float(np.max(np.abs(y - baseline)))
            if channels > 6 and max_vs_6 > 1e-5:
                raise AssertionError(f"Zero padding changed CPU output: Cin{channels}, {size}px, {max_vs_6}")
            entries.append({
                "size": size, "channels": channels,
                "model": model_path.name, "model_sha256": sha256(model_path),
                "input": input_path.name, "input_sha256": sha256(input_path),
                "reference": reference_path.name, "reference_sha256": sha256(reference_path),
                "input_shape": list(x.shape), "output_shape": list(y.shape),
                "cpu_max_abs_vs_cin6": max_vs_6,
            })
            print(prefix, max_vs_6, flush=True)
    (OUT / "conv-channel-repro.json").write_text(json.dumps({
        "source_model": "decoder_first_conv_repro.onnx",
        "source_model_sha256": sha256(OUT / "decoder_first_conv_repro.onnx"),
        "source_input": "decoder_diag_39_reference.bin",
        "source_input_sha256": sha256(OUT / "decoder_diag_39_reference.bin"),
        "onnxruntime_cpu": ort.__version__,
        "entries": entries,
    }, indent=2))


if __name__ == "__main__":
    main()
