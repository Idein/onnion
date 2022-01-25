import numpy as np
from onnion_runtime import Or

from .utils import check


def test_or_00():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    y = (np.random.randn(3, 4, 5) > 0).astype(bool)
    inputs = [x, y]

    check(Or, opset_version, attrs, inputs)


def test_or_01():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    y = (np.random.randn(5) > 0).astype(bool)
    inputs = [x, y]

    check(Or, opset_version, attrs, inputs)


def test_or_02():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    y = (np.random.randn(5, 6) > 0).astype(bool)
    inputs = [x, y]

    check(Or, opset_version, attrs, inputs)


def test_or_03():
    opset_version = 13
    attrs = dict()

    x = (np.random.randn(1, 4, 1, 6) > 0).astype(bool)
    y = (np.random.randn(3, 1, 5, 6) > 0).astype(bool)
    inputs = [x, y]

    check(Or, opset_version, attrs, inputs)
