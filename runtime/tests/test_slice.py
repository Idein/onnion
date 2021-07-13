import numpy as np
from onnion_runtime import Slice

from .utils import check


def test_slice_00():
    opset_version = 9
    starts = [0, 0]
    ends = [3, 10]
    axes = [0,1]
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version, starts=starts, ends=ends, axes=axes).run(v0)
    check("Slice", {"starts": starts, "ends": ends, "axes": axes}, [v0], outputs, opset_version)

def test_slice_01():
    opset_version = 9
    starts = [0, 0,3]
    ends = [3, 10, 4]
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version, starts=starts, ends=ends).run(v0)
    check("Slice", {"starts": starts, "ends": ends}, [v0], outputs, opset_version)

def test_slice_02():
    opset_version = 13
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0,1], dtype=np.int64)
    steps = np.array([1,1], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes, steps)
    check("Slice", dict(), [v0, starts, ends, axes, steps], outputs, opset_version)

def test_slice_03():
    """
    default axes
    """
    opset_version = 13
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends)
    check("Slice", dict(), [v0, starts, ends], outputs, opset_version)

def test_slice_04():
    """
    default steps
    """
    opset_version = 13
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0,1,2], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes)
    check("Slice", dict(), [v0, starts, ends, axes], outputs, opset_version)

def test_slice_05():
    """
    end out of bounds
    """
    opset_version = 13
    starts = np.array([1], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes, steps)
    check("Slice", dict(), [v0, starts, ends, axes, steps], outputs, opset_version)

def test_slice_06():
    """
    neg
    """
    opset_version = 13
    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes, steps)
    check("Slice", dict(), [v0, starts, ends, axes, steps], outputs, opset_version)

def test_slice_07():
    """
    neg steps
    """
    opset_version = 13
    starts = np.array([20, 10, 4], dtype=np.int64)
    ends = np.array([0, 0, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes, steps)
    check("Slice", dict(), [v0, starts, ends, axes, steps], outputs, opset_version)

def test_slice_08():
    """
    neg axes
    """
    opset_version = 13
    starts = np.array([0,0,3], dtype=np.int64)
    ends = np.array([20,10,4], dtype=np.int64)
    axes = np.array([0,-2,-1], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes)
    check("Slice", dict(), [v0, starts, ends, axes], outputs, opset_version)

def test_slice_09():
    """
    start out of bounds
    """
    opset_version = 13
    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    v0 = np.random.randn(20, 10, 5).astype(np.float32)

    outputs = Slice(opset_version).run(v0, starts, ends, axes, steps)
    check("Slice", dict(), [v0, starts, ends, axes, steps], outputs, opset_version)

