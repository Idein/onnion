import numpy as np
from onnion_runtime import Gather

from .utils import check


def test_gather_00():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])

    inputs = [data, indices]
    check(Gather, opset_version, attrs, inputs)


def test_gather_01():
    opset_version = 13
    axis = 1
    attrs = {"axis": axis}
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])

    inputs = [data, indices]
    check(Gather, opset_version, attrs, inputs)


def test_gather_02():
    opset_version = 13
    axis = 1
    attrs = {"axis": axis}
    data = np.random.randn(3, 3).astype(np.float32)
    indices = np.array([[0, 2]])

    inputs = [data, indices]
    check(Gather, opset_version, attrs, inputs)


def test_gather_03():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}
    data = np.arange(10).astype(np.float32)
    indices = np.array([0, -9, -10])

    inputs = [data, indices]
    check(Gather, opset_version, attrs, inputs)
