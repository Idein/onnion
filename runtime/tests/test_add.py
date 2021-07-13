import numpy as np
from onnion_runtime import Add

from .utils import check


def test_add_00():
    opset_version = 13
    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)

    inputs = [v0, v1]
    outputs = Add(opset_version).run(v0, v1)
    check("Add", dict(), inputs, outputs, opset_version)


def test_add_01():
    opset_version = 13
    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(3).astype(np.float32)

    inputs = [v0, v1]
    outputs = Add(opset_version).run(v0, v1)
    check("Add", dict(), inputs, outputs, opset_version)
