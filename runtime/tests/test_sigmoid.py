import numpy as np
from onnion_runtime import Sigmoid

from .utils import check


def test_sigmoid_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Sigmoid(opset_version).run(x)

    check("Sigmoid", dict(), [x], outputs, opset_version)
