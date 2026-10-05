from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def zero_policy(tmp_path_factory) -> Path:
    """A 61 -> 14 ONNX policy that always outputs zeros (= hold the default
    pose). Deterministic: the robot should stand and never walk."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.zeros((61, 14), dtype=np.float32), name="W")
    node = helper.make_node("MatMul", ["obs", "W"], ["actions"])
    graph = helper.make_graph(
        [node],
        "zero_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path_factory.mktemp("policies") / "zero.onnx"
    onnx.save(model, path)
    return path
