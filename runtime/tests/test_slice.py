import numpy as np
from onnion_runtime import Slice

from .utils import check


def test_slice_00():
    opset_version = 9
    starts = [0, 0]
    ends = [3, 10]
    axes = [0, 1]
    attrs = {"starts": starts, "ends": ends, "axes": axes}

    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x]

    check(Slice, opset_version, attrs, inputs)


def test_slice_01():
    opset_version = 9
    starts = [0, 0, 3]
    ends = [3, 10, 4]
    attrs = {"starts": starts, "ends": ends}

    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x]

    check(Slice, opset_version, attrs, inputs)


def test_slice_02():
    opset_version = 13
    attrs = dict()

    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes, steps]

    check(Slice, opset_version, attrs, inputs)


def test_slice_03():
    """
    default axes
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends]

    check(Slice, opset_version, attrs, inputs)


def test_slice_04():
    """
    default steps
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes]

    check(Slice, opset_version, attrs, inputs)


def test_slice_05():
    """
    end out of bounds
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([1], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes, steps]

    check(Slice, opset_version, attrs, inputs)


def test_slice_06():
    """
    neg
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes, steps]

    check(Slice, opset_version, attrs, inputs)


def test_slice_07():
    """
    neg steps
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([20, 10, 4], dtype=np.int64)
    ends = np.array([0, 0, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes, steps]

    check(Slice, opset_version, attrs, inputs)


def test_slice_08():
    """
    neg axes
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, -2, -1], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes]

    check(Slice, opset_version, attrs, inputs)


def test_slice_09():
    """
    start out of bounds
    """
    opset_version = 13
    attrs = dict()

    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    x = np.random.randn(20, 10, 5).astype(np.float32)
    inputs = [x, starts, ends, axes, steps]

    check(Slice, opset_version, attrs, inputs)
