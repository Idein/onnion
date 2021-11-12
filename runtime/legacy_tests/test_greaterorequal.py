import numpy as np
from onnion_runtime import GreaterOrEqual

from .utils import check


def test_greaterorequal_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x, y]
    outputs = GreaterOrEqual(opset_version).run(x, y)

    check("GreaterOrEqual", dict(), inputs, outputs, opset_version)


def test_greaterorequal_01():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    inputs = [x, y]
    outputs = GreaterOrEqual(opset_version).run(x, y)

    check("GreaterOrEqual", dict(), inputs, outputs, opset_version)
