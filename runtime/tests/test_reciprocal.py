import numpy as np
from onnion_runtime import Reciprocal

from .utils import check


def test_reciprocal_00():
    opset_version = 13
    attrs = dict()

    x = np.array([-4, 2]).astype(np.float32)
    inputs = [x]

    check(Reciprocal, opset_version, attrs, inputs)


def test_reciprocal_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 2, 3, 4).astype(np.float32)
    inputs = [x]

    check(Reciprocal, opset_version, attrs, inputs)
