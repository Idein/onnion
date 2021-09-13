import numpy as np
from onnion_runtime import ReduceMax

from .utils import check


def test_reducemax_00():
    opset_version = 13

    keepdims = 1
    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMax(opset_version, keepdims=keepdims).run(v0)

    check("ReduceMax", {"keepdims": keepdims}, [v0], outputs, opset_version)


def test_reducemax_01():
    opset_version = 13

    axes = [1]
    keepdims = 0
    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMax(opset_version, axes=axes, keepdims=keepdims).run(v0)

    check("ReduceMax", {"axes": axes, "keepdims": keepdims}, [v0], outputs, opset_version)


def test_reducemax_02():
    opset_version = 13

    axes = [1]
    keepdims = 1
    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMax(opset_version, axes=axes, keepdims=keepdims).run(v0)

    check("ReduceMax", {"axes": axes, "keepdims": keepdims}, [v0], outputs, opset_version)


def test_reducemax_03():
    opset_version = 13

    axes = [-2]
    keepdims = 1
    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMax(opset_version, axes=axes, keepdims=keepdims).run(v0)

    check("ReduceMax", {"axes": axes, "keepdims": keepdims}, [v0], outputs, opset_version)
