import numpy as np
from onnion_runtime import Sqrt

from .utils import check


def test_sqrt_00():
    opset_version = 14
    attrs = dict()

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Sqrt, opset_version, attrs, inputs)


def test_sqrt_01():
    opset_version = 14
    attrs = dict()

    x = np.array([1, 4, 9]).astype(np.float32)
    inputs = [x]

    check(Sqrt, opset_version, attrs, inputs)
