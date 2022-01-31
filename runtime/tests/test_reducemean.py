import numpy as np
from onnion_runtime import ReduceMean

from .utils import check


def test_reducemean_00():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceMean, opset_version, attrs, inputs)


def test_reducemean_01():
    opset_version = 13
    axes = [1]
    keepdims = 0
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceMean, opset_version, attrs, inputs)


def test_reducemean_02():
    opset_version = 13
    axes = [1]
    keepdims = 1
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceMean, opset_version, attrs, inputs)


def test_reducemean_03():
    opset_version = 13
    axes = [-2]
    keepdims = 1
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceMean, opset_version, attrs, inputs)
