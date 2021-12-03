import numpy as np
from onnion_runtime import GatherElements

from .utils import check


def test_gatherelements_00():
    opset_version = 13
    axis = 1
    attrs = {"axis": axis}

    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int32)
    inputs = [data, indices]

    check(GatherElements, opset_version, attrs, inputs)


def test_gatherelements_01():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[1, 2, 0], [2, 0, 0]], dtype=np.int32)
    inputs = [data, indices]

    check(GatherElements, opset_version, attrs, inputs)


def test_gatherelements_02():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[-1, -2, 0], [-2, 0, 0]], dtype=np.int32)
    inputs = [data, indices]

    check(GatherElements, opset_version, attrs, inputs)
