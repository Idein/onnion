import numpy as np
import onnx
from onnion_runtime import Add, Identity, Loop, Slice, Unsqueeze

from .utils import check


class SubGraph0:
    def __init__(self, opset_version):
        self.version = opset_version

    def run(self, iter_count, cond_in, y_in):
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        one = np.array(1).astype(np.int64)
        [end] = Add(self.version).run(iter_count, one)
        [slice_start] = Unsqueeze(self.version, axes=[0]).run(iter_count)
        [slice_end] = Unsqueeze(self.version, axes=[0]).run(end)
        [slice_out] = Slice(self.version).run(x, slice_start, slice_end)
        [y_out] = Add(self.version).run(y_in, slice_out)
        [cond_out] = Identity(self.version).run(cond_in)
        [scan_out] = Identity(self.version).run(y_out)
        return [cond_out, y_out, scan_out]


def test_loop_00():
    opset_version = 11

    y_in = onnx.helper.make_tensor_value_info("y_in", onnx.TensorProto.FLOAT, [1])
    y_out = onnx.helper.make_tensor_value_info("y_out", onnx.TensorProto.FLOAT, [1])
    scan_out = onnx.helper.make_tensor_value_info("scan_out", onnx.TensorProto.FLOAT, [1])
    cond_in = onnx.helper.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info("iter_count", onnx.TensorProto.INT64, [])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([-2]).astype(np.float32)

    x_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["x"],
        value=onnx.helper.make_tensor(
            name="const_tensor_x",
            data_type=onnx.TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        ),
    )

    one_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one"],
        value=onnx.helper.make_tensor(name="const_tensor_one", data_type=onnx.TensorProto.INT64, dims=(), vals=[1]),
    )

    i_add_node = onnx.helper.make_node("Add", inputs=["iter_count", "one"], outputs=["end"])

    start_unsqueeze_node = onnx.helper.make_node("Unsqueeze", inputs=["iter_count"], outputs=["slice_start"], axes=[0])

    end_unsqueeze_node = onnx.helper.make_node("Unsqueeze", inputs=["end"], outputs=["slice_end"], axes=[0])

    slice_node = onnx.helper.make_node("Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"])

    y_add_node = onnx.helper.make_node("Add", inputs=["y_in", "slice_out"], outputs=["y_out"])

    identity_node = onnx.helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

    scan_identity_node = onnx.helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

    loop_body = onnx.helper.make_graph(
        [
            identity_node,
            x_const_node,
            one_const_node,
            i_add_node,
            start_unsqueeze_node,
            end_unsqueeze_node,
            slice_node,
            y_add_node,
            scan_identity_node,
        ],
        "loop_body",
        [iter_count, cond_in, y_in],
        [cond_out, y_out, scan_out],
    )

    trip_count = np.array(5).astype(np.int64)
    cond = np.array(1).astype(bool)

    inputs = [trip_count, cond, y]
    outputs = Loop(opset_version, body=SubGraph0(opset_version)).run(trip_count, cond, y)

    check("Loop", {"body": loop_body}, inputs, outputs, opset_version)
