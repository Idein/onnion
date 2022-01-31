import numpy as np
from onnion_runtime import ReduceL1

from .utils import check


def test_reducel1_00():
    opset_version = 13
    attrs = {"keepdims": 1}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceL1, opset_version, attrs, inputs)


def test_reducel1_01():
    opset_version = 13
    attrs = {"keepdims": 0}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceL1, opset_version, attrs, inputs)


def test_reducel1_02():
    opset_version = 13
    attrs = {"keepdims": 1, "axes": [2]}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceL1, opset_version, attrs, inputs)


def test_reducel1_03():
    opset_version = 13
    attrs = {"keepdims": 1, "axes": [-2]}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceL1, opset_version, attrs, inputs)
