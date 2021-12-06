import numpy as np
from onnion_runtime import LeakyRelu

from .utils import check


def test_leakyrelu_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(LeakyRelu, opset_version, attrs, inputs)


def test_leakyrelu_01():
    opset_version = 13
    alpha = 2.0
    attrs = {"alpha": alpha}

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(LeakyRelu, opset_version, attrs, inputs)
