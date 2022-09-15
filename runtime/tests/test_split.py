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


def test_split_05():
    opset_version = 7
    axis = 2
    split = [12]
    attrs = {"axis": axis, "split": split}

    x = np.random.randn(1, 3, 12, 20).astype(np.float32)
    inputs = [x]

    check(Split, opset_version, attrs, inputs)


def test_split_06():
    opset_version = 7
    axis = 0
    split = [0, 0, 0]
    attrs = {"axis": axis, "split": split}

    x = np.array([]).astype(np.float32)

    inputs = [x]

    check(Split, opset_version, attrs, inputs)


def test_split_07():
    opset_version = 14
    axis = 0
    attrs = {"axis": axis}

    x = np.array([]).astype(np.float32)
    split = np.array([0, 0, 0]).astype(np.int64)

    inputs = [x, split]

    check(Split, opset_version, attrs, inputs)


def test_split_08():
    opset_version = 13
    axis = 1
    attrs = {"axis": axis}
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]).astype(np.float32)
    split = np.array([2, 4]).astype(np.int64)
    inputs = [x, split]

    check(Split, opset_version, attrs, inputs)


def test_split_09():
    opset_version = 12
    axis = 1
    split = [2, 4]
    attrs = {"axis": axis, "split": split}
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]).astype(np.float32)

    inputs = [x]

    check(Split, opset_version, attrs, inputs)


def test_split_10():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    split = np.array([2, 4]).astype(np.int64)
    inputs = [x, split]

    check(Split, opset_version, attrs, inputs)


def test_split_11():
    opset_version = 5
    axis = 0
    split = [2, 4]
    attrs = {"axis": axis, "split": split}
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    inputs = [x]

    check(Split, opset_version, attrs, inputs)
