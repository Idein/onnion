import numpy as np
from onnion_runtime import InstanceNormalization

from .utils import check


def test_instancenormalization_00():
    opset_version = 13

    x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
    s = np.array([1.0, 1.5]).astype(np.float32)
    bias = np.array([0, 1]).astype(np.float32)
    inputs = [x, s, bias]
    outputs = InstanceNormalization(opset_version).run(x, s, bias)

    check("InstanceNormalization", dict(), inputs, outputs, opset_version)


def test_instancenormalization_01():
    opset_version = 13

    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    epsilon = 1e-2
    inputs = [x, s, bias]
    outputs = InstanceNormalization(opset_version, epsilon=epsilon).run(x, s, bias)

    check("InstanceNormalization", {"epsilon": epsilon}, inputs, outputs, opset_version)
