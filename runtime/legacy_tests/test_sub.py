import numpy as np
from onnion_runtime import Sub

from .utils import check


def test_sub_00():
    opset_version = 13
    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)

    inputs = [v0, v1]
    outputs = Sub(opset_version).run(v0, v1)
    check("Sub", dict(), inputs, outputs, opset_version)


def test_sub_01():
    opset_version = 13
    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(3).astype(np.float32)

    inputs = [v0, v1]
    outputs = Sub(opset_version).run(v0, v1)
    check("Sub", dict(), inputs, outputs, opset_version)
