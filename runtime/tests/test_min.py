import numpy as np
from onnion_runtime import Min

from .utils import check


def test_min_00():
    opset_version = 13
    attrs = dict()

    v0 = np.array([3, 2, 1]).astype(np.float32)
    v1 = np.array([1, 4, 4]).astype(np.float32)
    v2 = np.array([2, 5, 0]).astype(np.float32)
    inputs = [v0, v1, v2]

    check(Min, opset_version, attrs, inputs)


def test_min_01():
    opset_version = 13
    attrs = dict()

    v0 = np.array([3, 2, 1]).astype(np.float32)
    v1 = np.array([1, 4, 4]).astype(np.float32)
    inputs = [v0, v1]

    check(Min, opset_version, attrs, inputs)


def test_min_02():
    opset_version = 13
    attrs = dict()

    v0 = np.array([3, 2, 1]).astype(np.float32)
    inputs = [v0]

    check(Min, opset_version, attrs, inputs)
