"""Stream-verify every base initializer against the browser external-data pair.

The inline ONNX file is nearly 2 GiB. This parser only reads protobuf field
headers and hashes each TensorProto.raw_data range in small chunks; it never
materializes the inline graph or all weights in memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx

CHUNK = 4 * 1024 * 1024


def varint(stream) -> int:
    value = 0
    for shift in range(0, 70, 7):
        byte = stream.read(1)
        if not byte:
            raise EOFError("Truncated protobuf varint")
        value |= (byte[0] & 127) << shift
        if not byte[0] & 128:
            return value
    raise ValueError("Invalid protobuf varint")


def fields(stream, end):
    while stream.tell() < end:
        key = varint(stream)
        number, wire = key >> 3, key & 7
        if wire == 2:
            length = varint(stream)
            start = stream.tell()
            if start + length > end:
                raise ValueError("Protobuf field exceeds parent message")
            yield number, wire, start, length, None
            stream.seek(start + length)
        elif wire == 0:
            yield number, wire, None, None, varint(stream)
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            if stream.tell() + length > end:
                raise ValueError("Truncated fixed-width protobuf field")
            stream.seek(length, 1)
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
    if stream.tell() != end:
        raise ValueError("Protobuf message boundary mismatch")


def range_hash(stream, start: int, length: int) -> str:
    stream.seek(start)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = stream.read(min(CHUNK, remaining))
        if not block:
            raise EOFError("Truncated ONNX tensor or external data")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def inline_initializers(path: Path) -> dict:
    tensors = {}
    with path.open("rb") as stream:
        end = path.stat().st_size
        graph_seen = False
        for number, wire, start, length, _ in fields(stream, end):
            if number != 7 or wire != 2:  # ModelProto.graph
                continue
            if graph_seen:
                raise ValueError("Duplicate ONNX graph")
            graph_seen = True
            for graph_field, graph_wire, graph_start, graph_len, _ in fields(stream, start + length):
                if graph_field != 5 or graph_wire != 2:  # GraphProto.initializer
                    continue
                name = None
                raw = None
                dims = []
                data_type = None
                for tensor_field, tensor_wire, tensor_start, tensor_len, value in fields(stream, graph_start + graph_len):
                    if tensor_field == 8 and tensor_wire == 2:  # TensorProto.name
                        stream.seek(tensor_start)
                        name = stream.read(tensor_len).decode("utf-8")
                    elif tensor_field == 9 and tensor_wire == 2:  # raw_data
                        raw = (tensor_len, range_hash(stream, tensor_start, tensor_len))
                    elif tensor_field == 1 and tensor_wire == 0:
                        dims.append(value)
                    elif tensor_field == 1 and tensor_wire == 2:  # packed dims
                        stream.seek(tensor_start)
                        while stream.tell() < tensor_start + tensor_len:
                            dims.append(varint(stream))
                    elif tensor_field == 2 and tensor_wire == 0:
                        data_type = value
                if not name or raw is None or data_type is None:
                    raise ValueError("Inline initializer missing name, raw_data or type")
                if name in tensors:
                    raise ValueError(f"Duplicate initializer {name}")
                tensors[name] = {"length": raw[0], "sha256": raw[1], "dims": dims, "data_type": data_type}
        if not graph_seen:
            raise ValueError("ONNX graph not found")
    return tensors


def verify_pair(inline: Path, model_path: Path, data_path: Path) -> dict:
    expected = inline_initializers(inline)
    external_model = onnx.load(str(model_path), load_external_data=False)
    if len(external_model.graph.initializer) != len(expected):
        raise AssertionError("Initializer count changed")
    counts = {"external": 0, "embedded": 0}
    spans = []
    with data_path.open("rb") as data:
        for tensor in external_model.graph.initializer:
            original = expected.pop(tensor.name, None)
            if original is None:
                raise AssertionError(f"Unexpected initializer {tensor.name}")
            if list(tensor.dims) != original["dims"] or tensor.data_type != original["data_type"]:
                raise AssertionError(f"Initializer shape/type changed: {tensor.name}")
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                info = {item.key: item.value for item in tensor.external_data}
                if info.get("location") != data_path.name:
                    raise AssertionError(f"External path mismatch: {tensor.name}")
                offset, length = int(info["offset"]), int(info["length"])
                if offset < 0 or length < 0 or offset + length > data_path.stat().st_size:
                    raise AssertionError(f"External offset/length invalid: {tensor.name}")
                digest = range_hash(data, offset, length)
                spans.append((offset, offset + length))
                counts["external"] += 1
            else:
                length = len(tensor.raw_data)
                digest = hashlib.sha256(tensor.raw_data).hexdigest()
                counts["embedded"] += 1
            if length != original["length"] or digest != original["sha256"]:
                raise AssertionError(f"Initializer bytes differ: {tensor.name}")
    if expected:
        raise AssertionError(f"Missing initializers: {list(expected)[:3]}")
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise AssertionError("External initializer ranges have a gap or overlap")
        position = end
    if position != data_path.stat().st_size:
        raise AssertionError("External data has unreferenced bytes")
    return {"method": "streamed SHA-256 per initializer versus inline raw_data",
            "all_initializer_values_match_inline": True,
            "initializer_count": sum(counts.values()), "external_count": counts["external"],
            "embedded_count": counts["embedded"], "external_bytes": position,
            "peak_tensor_read_chunk_bytes": CHUNK}


if __name__ == "__main__":
    root = Path("E:/TerrainDiffusionRuntime/webgpu-models")
    print(json.dumps(verify_pair(root / "base_model.onnx", root / "base_model_external.onnx",
                                 root / "base_model_external.data"), indent=2))
