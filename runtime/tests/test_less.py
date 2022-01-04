import numpy as np
from onnion_runtime import Less

from .utils import check


def test_less_00():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    inputs = [x, y]

    check(Less, opset_version, attrs, inputs)


def test_less_01():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    y = (np.random.randn(5) * 10).astype(np.int32)
    inputs = [x, y]

    check(Less, opset_version, attrs, inputs)
