import numpy as np
from onnion_runtime import BitShift

from .utils import check


def test_bitshift_00():
    opset_version = 13
    direction = "LEFT"
    attrs = {"direction": direction}

    x = np.array([16, 4, 1]).astype(np.uint64)
    y = np.array([1, 2, 3]).astype(np.uint64)
    inputs = [x, y]

    check(BitShift, opset_version, attrs, inputs)


def test_bitshift_01():
    opset_version = 13
    direction = "RIGHT"
    attrs = {"direction": direction}

    x = np.array([16, 4, 1]).astype(np.uint64)
    y = np.array([1, 2, 3]).astype(np.uint64)
    inputs = [x, y]

    check(BitShift, opset_version, attrs, inputs)
