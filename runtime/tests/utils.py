import inspect
import os
import platform
import tempfile
from typing import Any, Dict, List, Union

import numpy as np

try:
    import onnx
    import onnxruntime
    from onnx import checker, helper, mapping

    WITHOUT_ONNXRUNTIME = False
except Exception:
    WITHOUT_ONNXRUNTIME = True


def on_arm32():
    try:
        result = bool(int(os.environ["ONNION_TEST_ON_ARM32"]))
    except Exception:
        arch = platform.machine()
        if arch == "x86_64":
            result = False
        elif arch == "armv7l":
            result = True
        else:
            raise Exception("on_arm32: unknown arch")

    return result


def _load_data(f):
    return list(np.load(f, allow_pickle=True))


def _save_data(f, vs):
    np.save(f, np.array(vs, dtype=object))


def load_test_data():
    caller_name = inspect.stack()[1].function
    file_name = f"tests/{caller_name}.npy"
    with open(file_name, "rb") as f:
        inputs = _load_data(f)
        outputs = _load_data(f)
    return inputs, outputs


def save_test_data(inputs, outputs):
    caller_name = inspect.stack()[1].function
    file_name = f"tests/{caller_name}.npy"
    with open(file_name, "wb") as f:
        _save_data(f, inputs)
        _save_data(f, outputs)


def check_by_data(expected, result, max_error=1e-4):
    assert len(expected) == len(result)
    for a, b in zip(expected, result):
        if a.dtype == bool:
            assert np.all(a == b)
        else:
            assert np.all(abs(a - b) < max_error)


def _convert_type(dtype):
    assert not WITHOUT_ONNXRUNTIME
    return mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]


def _run_onnx(model, inputs, output_names):
    assert not WITHOUT_ONNXRUNTIME
    checker.check_model(model)
    with tempfile.NamedTemporaryFile(mode="w") as f:
        onnx.save(model, f.name)
        sess = onnxruntime.InferenceSession(f.name)
        return sess.run(output_names, inputs)


def check_by_onnxruntime(
    op_name: str,
    attrs: Dict[str, Any],
    input_values: List[Union[np.array, List[np.array]]],
    output_values: List[Union[np.array, List[np.array]]],
    opset_version: int,
    max_error=1e-4,
) -> List[Union[np.array, List[np.array]]]:
    assert not WITHOUT_ONNXRUNTIME

    input_names = [f"input{i}" for i, _ in enumerate(input_values)]
    output_names = [f"output{i}" for i, _ in enumerate(output_values)]
    node = helper.make_node(op_name, input_names, output_names, **attrs)

    input_tensors = []
    for n, v in zip(input_names, input_values):
        if type(v) == list:
            input_tensors.append(helper.make_sequence_value_info(n, _convert_type(v[0].dtype), list(v[0].shape)))
        else:
            input_tensors.append(helper.make_tensor_value_info(n, _convert_type(v.dtype), list(v.shape)))

    output_tensors = []
    for n, v in zip(output_names, output_values):
        if type(v) == list:
            output_tensors.append(helper.make_sequence_value_info(n, _convert_type(v[0].dtype), list(v[0].shape)))
        else:
            output_tensors.append(helper.make_tensor_value_info(n, _convert_type(v.dtype), list(v.shape)))

    graph = helper.make_graph([node], "test_graph", input_tensors, output_tensors)
    opset_imports = [helper.make_opsetid("", opset_version)]
    model = helper.make_model(graph, opset_imports=opset_imports)

    inputs = dict()
    for n, v in zip(input_names, input_values):
        inputs[n] = v
    results = _run_onnx(model, inputs, output_names)

    check_by_data(results, output_values, max_error)
    return results
