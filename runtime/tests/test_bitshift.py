import numpy as np
from onnion_runtime import BitShift

from .utils import check


def test_bitshift_00():
    opset_version = 13

    direction = "LEFT"
    x = np.array([16, 4, 1]).astype(np.uint64)
    y = np.array([1, 2, 3]).astype(np.uint64)
    inputs = [x, y]
    outputs = BitShift(opset_version, direction=direction).run(x, y)

    check("BitShift", {"direction": direction}, inputs, outputs, opset_version)


def test_bitshift_01():
    opset_version = 13

    direction = "RIGHT"
    x = np.array([16, 4, 1]).astype(np.uint64)
    y = np.array([1, 2, 3]).astype(np.uint64)
    inputs = [x, y]
    outputs = BitShift(opset_version, direction=direction).run(x, y)

    check("BitShift", {"direction": direction}, inputs, outputs, opset_version)
