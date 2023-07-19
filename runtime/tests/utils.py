import inspect
import logging
import os
import platform
import tempfile
from typing import Any, Dict, List, Union

import numpy as np

try:
    import onnx
    import onnxruntime
    from onnx import checker, helper, numpy_helper

    WITHOUT_ONNXRUNTIME = False
except Exception:
    WITHOUT_ONNXRUNTIME = True

LOGGER = logging.getLogger(__name__)


def on_arm32():
    try:
        result = bool(int(os.environ["ONNION_TEST_ON_ARM32"]))
    except Exception:
        arch = platform.machine()
        if arch == "x86_64" or arch == "arm64":
            result = False
        elif arch == "armv7l":
            result = True
        else:
            raise Exception("on_arm32: unknown arch")

    LOGGER.info(f"on_arm32: {result}")
    return result


def _load_data(f, vs):
    if vs is None:
        return None
    elif type(vs) == list:
        res = []
        for v in vs:
            res.append(_load_data(f, v))
        return res
    else:
        return np.load(f)


def load_test_data(file_name, vs):
    LOGGER.info(f"load from {file_name}")
    with open(file_name, "rb") as f:
        res = _load_data(f, vs)
    return res


def _save_data(f, vs):
    if vs is None:
        pass
    elif type(vs) == list:
        for v in vs:
            _save_data(f, v)
    else:
        np.save(f, vs)


def save_test_data(file_name, vs):
    LOGGER.info(f"save to {file_name}")
    with open(file_name, "wb") as f:
        _save_data(f, vs)


def check_by_data(expected, result, max_error=1e-4):
    assert len(expected) == len(result)
    for a, b in zip(expected, result):
        if a.dtype == bool:
            assert np.all(a == b)
        else:
            assert np.all(abs(a - b) < max_error)


def _convert_type(dtype):
    assert not WITHOUT_ONNXRUNTIME
    return helper.np_dtype_to_tensor_dtype(dtype)


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

    input_names = []
    for i, v in enumerate(input_values):
        if v is None:
            input_names.append("")
        else:
            input_names.append(f"input{i}")
    output_names = [f"output{i}" for i, _ in enumerate(output_values)]
    if op_name in ["Constant", "ConstantOfShape"]:
        attrs["value"] = numpy_helper.from_array(attrs["value"])
    elif op_name == "If":
        attrs["then_branch"] = attrs["then_branch"].to_onnx()
        attrs["else_branch"] = attrs["else_branch"].to_onnx()
    elif op_name == "Loop":
        attrs["body"] = attrs["body"].to_onnx()
    node = helper.make_node(op_name, input_names, output_names, **attrs)

    input_tensors = []
    for n, v in zip(input_names, input_values):
        if v is None:
            pass
        elif type(v) == list:
            input_tensors.append(helper.make_tensor_sequence_value_info(n, _convert_type(v[0].dtype), list(v[0].shape)))
        else:
            input_tensors.append(helper.make_tensor_value_info(n, _convert_type(v.dtype), list(v.shape)))

    output_tensors = []
    for n, v in zip(output_names, output_values):
        if type(v) == list:
            output_tensors.append(helper.make_tensor_sequence_value_info(n, _convert_type(v[0].dtype), list(v[0].shape)))
        else:
            output_tensors.append(helper.make_tensor_value_info(n, _convert_type(v.dtype), list(v.shape)))

    graph = helper.make_graph([node], "test_graph", input_tensors, output_tensors)
    opset_imports = [helper.make_opsetid("", opset_version)]
    model = helper.make_model(graph, opset_imports=opset_imports)

    inputs = dict()
    for n, v in zip(input_names, input_values):
        if v is not None:
            inputs[n] = v
    results = _run_onnx(model, inputs, output_names)

    check_by_data(results, output_values, max_error)
    return results


def check(onnion_op, opset_version, attrs, input_values, max_error=1e-4):
    caller_name = inspect.stack()[1].function
    input_npy_file = f"tests/{caller_name}_inputs.npy"
    output_npy_file = f"tests/{caller_name}_outputs.npy"

    op = onnion_op(opset_version, **attrs)

    if on_arm32():
        inputs = load_test_data(input_npy_file, input_values)
        outputs = op.run(*inputs)
        outputs0 = load_test_data(output_npy_file, outputs)
        check_by_data(outputs0, outputs, max_error)
    else:
        outputs = op.run(*input_values)
        op_name = type(op).__name__
        outputs0 = check_by_onnxruntime(op_name, attrs, input_values, outputs, opset_version, max_error)
        save_test_data(input_npy_file, input_values)
        save_test_data(output_npy_file, outputs0)
