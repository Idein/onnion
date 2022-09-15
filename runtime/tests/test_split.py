import numpy as np
from onnion_runtime import Split

from .utils import check


def test_split_00():
    opset_version = 13
    axis = 4
    attrs = {"axis": axis}

    x = np.random.randn(1, 3, 20, 20, 8).astype(np.float32)
    split = np.array([2, 2, 4]).astype(np.int64)
    inputs = [x, split]

    check(Split, opset_version, attrs, inputs)


def test_split_01():
    opset_version = 11
    axis = 4
    split = [2, 2, 4]
    attrs = {"axis": axis, "split": split}
    x = np.random.randn(1, 3, 20, 20, 8).astype(np.float32)
    inputs = [x]

    check(Split, opset_version, attrs, inputs)


def test_split_02():
    opset_version = 14
    attrs = {}

    x = np.random.randn(10, 3, 3, 20).astype(np.float32)
    split = np.array([2, 4, 4]).astype(np.int64)
    inputs = [x, split]

    check(Split, opset_version, attrs, inputs)


def test_split_03():
    opset_version = 14
    axis = 2
    attrs = {"axis": axis}

    x = np.random.randn(1, 3, 12, 20).astype(np.float32)
    split = np.array([3, 3, 6]).astype(np.int64)
    inputs = [x, split]

    check(Split, opset_version, attrs, inputs)


def test_split_04():
    opset_version = 7
    axis = 2
    split = [3, 3, 6]
    attrs = {"axis": axis, "split": split}

    x = np.random.randn(1, 3, 12, 20).astype(np.float32)
    inputs = [x]

    check(Split, opset_version, attrs, inputs)
