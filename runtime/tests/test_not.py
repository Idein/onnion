import numpy as np
from onnion_runtime import Not

from .utils import check


def test_not_00():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4) > 0).astype(bool)
    inputs = [x]

    check(Not, opset_version, attrs, inputs)


def test_not_01():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    inputs = [x]

    check(Not, opset_version, attrs, inputs)


def test_not_02():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    inputs = [x]

    check(Not, opset_version, attrs, inputs)
