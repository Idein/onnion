import numpy as np
from onnion_runtime import Mean

from .utils import check


def test_mean_00():
    opset_version = 13
    attrs = dict()

    v0 = np.array([3, 0, 2]).astype(np.float32)
    v1 = np.array([1, 3, 4]).astype(np.float32)
    v2 = np.array([2, 6, 6]).astype(np.float32)
    inputs = [v0, v1, v2]

    check(Mean, opset_version, attrs, inputs)


def test_mean_01():
    opset_version = 13
    attrs = dict()

    v0 = np.array([3, 0, 2]).astype(np.float32)
    v1 = np.array([1, 3, 4]).astype(np.float32)
    inputs = [v0, v1]

    check(Mean, opset_version, attrs, inputs)


def test_mean_02():
    opset_version = 13
    attrs = dict()

    v0 = np.array([3, 0, 2]).astype(np.float32)
    inputs = [v0]

    check(Mean, opset_version, attrs, inputs)
