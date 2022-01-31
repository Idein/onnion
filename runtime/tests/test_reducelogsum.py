import numpy as np
from onnion_runtime import ReduceLogSum

from .utils import check


def test_reducelogsum_00():
    opset_version = 13
    attrs = {"keepdims": 1}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [np.abs(x)]

    check(ReduceLogSum, opset_version, attrs, inputs)


def test_reducelogsum_01():
    opset_version = 13
    attrs = {"keepdims": 0}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [np.abs(x)]

    check(ReduceLogSum, opset_version, attrs, inputs)


def test_reducelogsum_02():
    opset_version = 13
    attrs = {"keepdims": 1, "axes": [2, 1]}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [np.abs(x)]

    check(ReduceLogSum, opset_version, attrs, inputs)


def test_reducelogsum_03():
    opset_version = 13
    attrs = {"keepdims": 1, "axes": [-2]}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [np.abs(x)]

    check(ReduceLogSum, opset_version, attrs, inputs)
