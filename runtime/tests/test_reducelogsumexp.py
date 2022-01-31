import numpy as np
from onnion_runtime import ReduceLogSumExp

from .utils import check


def test_reducelogsumexp_00():
    opset_version = 13
    attrs = {"keepdims": 1}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceLogSumExp, opset_version, attrs, inputs)


def test_reducelogsumexp_01():
    opset_version = 13
    attrs = {"keepdims": 0}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceLogSumExp, opset_version, attrs, inputs)


def test_reducelogsumexp_02():
    opset_version = 13
    attrs = {"keepdims": 1, "axes": [2, 1]}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceLogSumExp, opset_version, attrs, inputs)


def test_reducelogsumexp_03():
    opset_version = 13
    attrs = {"keepdims": 1, "axes": [-2]}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceLogSumExp, opset_version, attrs, inputs)
