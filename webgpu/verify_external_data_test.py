"""Small tamper test for the streaming external-initializer verifier."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import onnx

from verify_external_data import verify_pair


class ExternalDataVerificationTest(unittest.TestCase):
    def test_detects_changed_external_initializer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["x"], ["y"])], "fixture",
                [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
                [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
                [onnx.numpy_helper.from_array(np.arange(512, dtype=np.float32), "large"),
                 onnx.numpy_helper.from_array(np.array([2], dtype=np.float32), "small")],
            )
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 17)])
            inline = root / "inline.onnx"
            external = root / "external.onnx"
            data = root / "external.data"
            onnx.save(model, str(inline))
            onnx.save_model(model, str(external), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=data.name,
                            size_threshold=1024, convert_attribute=False)
            self.assertTrue(verify_pair(inline, external, data)["all_initializer_values_match_inline"])
            with data.open("r+b") as stream:
                byte = stream.read(1)
                stream.seek(0)
                stream.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(AssertionError, "Initializer bytes differ"):
                verify_pair(inline, external, data)


if __name__ == "__main__":
    unittest.main()
