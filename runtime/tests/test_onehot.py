import numpy as np
from onnion_runtime import OneHot

from .utils import check


def test_onehot_00():
    opset_version = 13
    attrs = {"axis": 1}

    indices = np.array([[1, 9], [2, 4]], dtype=np.float32)
    depth = np.array([10], dtype=np.float32)
    values = np.array([1, 3], dtype=np.float32)
    inputs = [indices, depth, values]

    check(OneHot, opset_version, attrs, inputs)


def test_onehot_01():
    opset_version = 13
    attrs = {"axis": -2}

    indices = np.array([[1, 9], [2, 4]], dtype=np.float32)
    depth = np.array([10], dtype=np.float32)
    values = np.array([1, 3], dtype=np.float32)
    inputs = [indices, depth, values]

    check(OneHot, opset_version, attrs, inputs)


def test_onehot_02():
    opset_version = 13
    attrs = {"axis": 1}

    indices = np.array([0, -7, -8], dtype=np.int64)
    depth = np.array([10], dtype=np.float32)
    values = np.array([1, 3], dtype=np.float32)
    inputs = [indices, depth, values]

    check(OneHot, opset_version, attrs, inputs)


def test_onehot_03():
    opset_version = 13
    attrs = dict()

    indices = np.array([0, 7, 8], dtype=np.int64)
    depth = np.array([12], dtype=np.float32)
    values = np.array([1, 3], dtype=np.int32)
    inputs = [indices, depth, values]

    check(OneHot, opset_version, attrs, inputs)
