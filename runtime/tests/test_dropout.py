import numpy as np
from onnion_runtime import Dropout

from .utils import check


def test_dropout_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Dropout, opset_version, attrs, inputs)


def test_dropout_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.array(0.1).astype(np.float32)
    inputs = [x, r]

    check(Dropout, opset_version, attrs, inputs)


def test_dropout_02():
    opset_version = 11
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Dropout, opset_version, attrs, inputs)
