import tempfile

import numpy as np
import onnx
import onnxruntime
from onnx import checker, helper, mapping


def convert_type(dtype):
    return mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]


def run_onnx(model, inputs, outputs):
    checker.check_model(model)
    output_names = [k for k in outputs]
    with tempfile.NamedTemporaryFile(mode="w") as f:
        onnx.save(model, f.name)
        sess = onnxruntime.InferenceSession(f.name)
        return sess.run(output_names, inputs)


def check(op_name, attrs, inputs, outputs, opset_version, max_error=1e-4):
    node = helper.make_node(op_name, inputs.keys(), outputs.keys(), attrs)
    input_tensors = [helper.make_tensor_value_info(k, convert_type(inputs[k].dtype), list(inputs[k].shape)) for k in inputs]
    output_tensors = [helper.make_tensor_value_info(k, convert_type(outputs[k].dtype), list(outputs[k].shape)) for k in outputs]
    graph = helper.make_graph([node], "test_graph", input_tensors, output_tensors)
    opset_imports = [helper.make_opsetid("", opset_version)]
    model = helper.make_model(graph, opset_imports=opset_imports)

    results = run_onnx(model, inputs, outputs)
    for a, b in zip(results, [outputs[k] for k in outputs]):
        assert np.all(abs(a - b) < max_error)
