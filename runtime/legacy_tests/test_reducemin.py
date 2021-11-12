import numpy as np
from onnion_runtime import ReduceMin

from .utils import check


def test_reducemin_00():
    opset_version = 13
    axes = None
    keepdims = 1

    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMin(opset_version, axes=axes, keepdims=keepdims).run(v0)
    check("ReduceMin", {"keepdims": keepdims}, [v0], outputs, opset_version)


def test_reducemin_01():
    opset_version = 13
    axes = [1]
    keepdims = 0

    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMin(opset_version, axes=axes, keepdims=keepdims).run(v0)
    check("ReduceMin", {"axes": axes, "keepdims": keepdims}, [v0], outputs, opset_version)


def test_reducemin_02():
    opset_version = 13
    axes = [1]
    keepdims = 1

    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMin(opset_version, axes=axes, keepdims=keepdims).run(v0)
    check("ReduceMin", {"axes": axes, "keepdims": keepdims}, [v0], outputs, opset_version)


def test_reducemin_03():
    opset_version = 13
    axes = [-2]
    keepdims = 1

    v0 = np.random.randn(3, 2, 2).astype(np.float32)
    outputs = ReduceMin(opset_version, axes=axes, keepdims=keepdims).run(v0)
    check("ReduceMin", {"axes": axes, "keepdims": keepdims}, [v0], outputs, opset_version)
