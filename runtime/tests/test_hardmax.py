import numpy as np
from onnion_runtime import Hardmax

from .utils import check


def test_hardmax_00():
    opset_version = 13
    attrs = dict()

    x = np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)
    inputs = [x]

    check(Hardmax, opset_version, attrs, inputs)


def test_hardmax_01():
    opset_version = 13
    attrs = dict()

    x = np.array([[3, 3, 3, 1]]).astype(np.float32)
    inputs = [x]

    check(Hardmax, opset_version, attrs, inputs)


def test_hardmax_02():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Hardmax, opset_version, attrs, inputs)
