import numpy as np
from onnion_runtime import ReduceProd

from .utils import check


def test_reduceprod_00():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceProd, opset_version, attrs, inputs)


def test_reduceprod_01():
    opset_version = 13
    axes = [1]
    keepdims = 0
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceProd, opset_version, attrs, inputs)


def test_reduceprod_02():
    opset_version = 13
    axes = [1]
    keepdims = 1
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceProd, opset_version, attrs, inputs)


def test_reduceprod_03():
    opset_version = 13
    axes = [-2]
    keepdims = 1
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceProd, opset_version, attrs, inputs)
