import numpy as np
from onnion_runtime import Constant, If

from .utils import check


class SubGraph:
    def __init__(self, opset_version, v, prefix):
        self.version = opset_version
        self.value = v
        self.prefix = prefix

    def run(self):
        return Constant(self.version, value=self.value).run()

    def to_onnx(self):
        import onnx

        v = onnx.helper.make_tensor_value_info(f"{self.prefix}_out", onnx.TensorProto.FLOAT, list(self.value.shape))
        const_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=[f"{self.prefix}_out"], value=onnx.numpy_helper.from_array(self.value)
        )
        body = onnx.helper.make_graph([const_node], f"{self.prefix}_body", [], [v])
        return body


def test_if_00():
    opset_version = 13
    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)
    attrs = {"then_branch": SubGraph(opset_version, x, "then"), "else_branch": SubGraph(opset_version, y, "else")}

    cond = np.array(1).astype(bool)
    inputs = [cond]

    check(If, opset_version, attrs, inputs)


def test_if_01():
    opset_version = 13
    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)
    attrs = {"then_branch": SubGraph(opset_version, x, "then"), "else_branch": SubGraph(opset_version, y, "else")}

    cond = np.array(0).astype(bool)
    inputs = [cond]

    check(If, opset_version, attrs, inputs)
