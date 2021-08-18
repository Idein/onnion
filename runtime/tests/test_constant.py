import numpy as np
from onnion_runtime import Constant
from onnx import numpy_helper

from .utils import check


def test_constant_00():
    opset_version = 14

    v = np.random.randn(2, 3).astype(np.float32)
    outputs = Constant(opset_version, value=v).run()

    check("Constant", {"value": numpy_helper.from_array(v)}, [], outputs, opset_version)
