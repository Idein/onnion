import numpy as np
import onnx
from onnion_runtime import Constant, If

from .utils import check


class SubGraph0:
    def __init__(self, opset_version):
        self.version = opset_version

    def run(self):
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        return Constant(self.version, value=x).run()


class SubGraph1:
    def __init__(self, opset_version):
        self.version = opset_version

    def run(self):
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)
        return Constant(self.version, value=y).run()


def test_if_00():
    opset_version = 13

    then_out = onnx.helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_const_node = onnx.helper.make_node("Constant", inputs=[], outputs=["then_out"], value=onnx.numpy_helper.from_array(x))

    else_const_node = onnx.helper.make_node("Constant", inputs=[], outputs=["else_out"], value=onnx.numpy_helper.from_array(y))

    then_body = onnx.helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = onnx.helper.make_graph([else_const_node], "else_body", [], [else_out])

    cond = np.array(1).astype(bool)
    outputs = If(opset_version, then_branch=SubGraph0(opset_version), else_branch=SubGraph1(opset_version)).run(cond)

    check("If", {"then_branch": then_body, "else_branch": else_body}, [cond], outputs, opset_version)


def test_if_01():
    opset_version = 13

    then_out = onnx.helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_const_node = onnx.helper.make_node("Constant", inputs=[], outputs=["then_out"], value=onnx.numpy_helper.from_array(x))

    else_const_node = onnx.helper.make_node("Constant", inputs=[], outputs=["else_out"], value=onnx.numpy_helper.from_array(y))

    then_body = onnx.helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = onnx.helper.make_graph([else_const_node], "else_body", [], [else_out])

    cond = np.array(0).astype(bool)
    outputs = If(opset_version, then_branch=SubGraph0(opset_version), else_branch=SubGraph1(opset_version)).run(cond)

    check("If", {"then_branch": then_body, "else_branch": else_body}, [cond], outputs, opset_version)
