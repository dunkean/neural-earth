"""Extract early decoder graph prefixes for CPU/WebGPU divergence localization."""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[1] / "backend"))
from terrain_paths import RUNTIME_ROOT

from pathlib import Path
from copy import deepcopy
import json

import numpy as np
import onnx
import onnxruntime as ort

DEST = (RUNTIME_ROOT / 'webgpu-models')
SOURCE = DEST / "decoder_model.onnx"
FIXTURE = json.loads((DEST / "crop64-coast-manifest.json").read_text())
FEEDS = FIXTURE["first_real_forward"]["decoder_model"]


def load(name):
    spec = FEEDS[name]
    return np.fromfile(DEST / spec["file"], dtype=np.float32).reshape(spec["shape"])


def main():
    graph = onnx.load(str(SOURCE), load_external_data=False)
    data = {}
    for index in (39, 48, 49, 108, 256):
        node = graph.graph.node[index]
        name = node.output[0]
        model = DEST / f"decoder_diag_{index}.onnx"
        onnx.utils.extract_model(str(SOURCE), str(model), [value.name for value in graph.graph.input], [name])
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        feed = {item.name: load(item.name) for item in session.get_inputs()}
        output = session.run(None, feed)[0]
        path = DEST / f"decoder_diag_{index}_reference.bin"
        output.tofile(path)
        data[str(index)] = {"model": model.name, "reference": path.name, "output_name": name,
                            "output_shape": list(output.shape), "node": node.name, "op": node.op_type,
                            "min": float(output.min()), "max": float(output.max())}
        print(index, node.op_type, output.shape, model.stat().st_size, flush=True)
    frozen = onnx.load(str(DEST / "decoder_diag_49.onnx"))
    weight = np.fromfile(DEST / "decoder_diag_48_reference.bin", dtype=np.float32).reshape(64, 6, 3, 3)
    next(node for node in frozen.graph.node if node.name == "/model/512x512_conv/Conv").input[1] = "frozen_conv_weight"
    frozen.graph.initializer.append(onnx.numpy_helper.from_array(weight, "frozen_conv_weight"))
    frozen_path = DEST / "decoder_diag_49_frozen.onnx"
    onnx.save(frozen, str(frozen_path))
    data["49_frozen"] = {**data["49"], "model": frozen_path.name}
    conv = deepcopy(graph.graph.node[49])
    conv.input[0] = "x6"
    conv.input[1] = "weight"
    conv.output[0] = "y"
    repro = onnx.helper.make_model(onnx.helper.make_graph(
        [conv], "decoder-first-conv-repro",
        [onnx.helper.make_tensor_value_info("x6", onnx.TensorProto.FLOAT, [1, 6, 512, 512])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64, 512, 512])],
        [onnx.numpy_helper.from_array(weight, "weight")],
    ), opset_imports=list(graph.opset_import))
    repro.ir_version = graph.ir_version
    repro_path = DEST / "decoder_first_conv_repro.onnx"
    onnx.checker.check_model(repro)
    onnx.save(repro, str(repro_path))
    repro_cpu = ort.InferenceSession(str(repro_path), providers=["CPUExecutionProvider"])
    repro_y = repro_cpu.run(None, {"x6": np.fromfile(DEST / "decoder_diag_39_reference.bin", dtype=np.float32).reshape(1, 6, 512, 512)})[0]
    reference_y = np.fromfile(DEST / "decoder_diag_49_reference.bin", dtype=np.float32).reshape(1, 64, 512, 512)
    if float(np.max(np.abs(repro_y - reference_y))) > 1e-5:
        raise AssertionError("Single-Conv repro differs from decoder prefix")
    data["first_conv_repro"] = {
        "model": repro_path.name,
        "input": "decoder_diag_39_reference.bin",
        "input_shape": [1, 6, 512, 512],
        "reference": "decoder_diag_49_reference.bin",
        "output_shape": [1, 64, 512, 512],
        "cpu_max_abs_vs_decoder_prefix": float(np.max(np.abs(repro_y - reference_y))),
    }
    (DEST / "decoder-diagnostic.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
