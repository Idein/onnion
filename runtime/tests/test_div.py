import numpy as np
from onnion_runtime import Div

from .utils import check


def test_div_00():
    opset_version = 13
    attrs = dict()

    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)
    inputs = [v0, v1]

    check(Div, opset_version, attrs, inputs)


def test_div_01():
    opset_version = 13
    attrs = dict()

    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(3).astype(np.float32)
    inputs = [v0, v1]

    check(Div, opset_version, attrs, inputs)
