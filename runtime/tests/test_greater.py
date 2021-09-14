import numpy as np
from onnion_runtime import Greater

from .utils import check


def test_greater_00():
    opset_version = 13

    x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    inputs = [x, y]
    outputs = Greater(opset_version).run(x, y)

    check("Greater", dict(), inputs, outputs, opset_version)


def test_greater_01():
    opset_version = 13

    x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    y = (np.random.randn(5) * 10).astype(np.int32)
    inputs = [x, y]
    outputs = Greater(opset_version).run(x, y)

    check("Greater", dict(), inputs, outputs, opset_version)
