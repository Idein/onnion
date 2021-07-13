import tempfile
from typing import Any, Dict, List

import numpy as np
import onnx
import onnxruntime
from onnx import checker, helper, mapping


def convert_type(dtype):
    return mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]


def run_onnx(model, inputs, output_names):
    checker.check_model(model)
    with tempfile.NamedTemporaryFile(mode="w") as f:
        onnx.save(model, f.name)
        sess = onnxruntime.InferenceSession(f.name)
        return sess.run(output_names, inputs)


def check(
    op_name: str,
    attrs: Dict[str, Any],
    input_values: List[np.array],
    output_values: List[np.array],
    opset_version: int,
    max_error=1e-4,
):
    input_names = [f"input{i}" for i, _ in enumerate(input_values)]
    output_names = [f"output{i}" for i, _ in enumerate(output_values)]
    node = helper.make_node(op_name, input_names, output_names, **attrs)
    input_tensors = [
        helper.make_tensor_value_info(n, convert_type(v.dtype), list(v.shape)) for n, v in zip(input_names, input_values)
    ]
    output_tensors = [
        helper.make_tensor_value_info(n, convert_type(v.dtype), list(v.shape)) for n, v in zip(output_names, output_values)
    ]
    graph = helper.make_graph([node], "test_graph", input_tensors, output_tensors)
    opset_imports = [helper.make_opsetid("", opset_version)]
    model = helper.make_model(graph, opset_imports=opset_imports)

    inputs = dict()
    for n, v in zip(input_names, input_values):
        inputs[n] = v
    results = run_onnx(model, inputs, output_names)
    for a, b in zip(results, output_values):
        assert np.all(abs(a - b) < max_error)
