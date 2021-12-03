import numpy as np
from onnion_runtime import HardSigmoid

from .utils import check


def test_hardsigmoid_00():
    opset_version = 13
    alpha = 0.5
    beta = 0.6
    attrs = {"alpha": alpha, "beta": beta}

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(HardSigmoid, opset_version, attrs, inputs)


def test_hardsigmoid_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(HardSigmoid, opset_version, attrs, inputs)
