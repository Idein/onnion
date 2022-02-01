import numpy as np
import pytest
from onnion_runtime import ReduceSum

from .utils import check, on_arm32


def test_reducesum_00():
    opset_version = 11
    keepdims = 1
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_01():
    opset_version = 11
    axes = [1]
    keepdims = 0
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_02():
    opset_version = 11
    axes = [1]
    keepdims = 1
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_03():
    opset_version = 11
    axes = [-2]
    keepdims = 1
    attrs = {"axes": axes, "keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_04():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_05():
    opset_version = 13
    keepdims = 0
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    inputs = [x, axes]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_06():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    inputs = [x, axes]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_07():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    axes = np.array([-2], dtype=np.int64)
    inputs = [x, axes]

    check(ReduceSum, opset_version, attrs, inputs)


@pytest.mark.skipif(on_arm32(), reason="need to pass tests on x86_64")
@pytest.mark.xfail
def test_reducesum_08():
    opset_version = 13
    keepdims = 0
    noop_with_empty_axes = 1
    attrs = {"keepdims": keepdims, "noop_with_empty_axes": noop_with_empty_axes}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    inputs = [x]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_09():
    opset_version = 13
    keepdims = 1
    noop_with_empty_axes = 1
    attrs = {"keepdims": keepdims, "noop_with_empty_axes": noop_with_empty_axes}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    inputs = [x, axes]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_10():
    opset_version = 13
    keepdims = 1
    noop_with_empty_axes = 1
    attrs = {"keepdims": keepdims, "noop_with_empty_axes": noop_with_empty_axes}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    axes = np.array([], dtype=np.int64)
    inputs = [x, axes]

    check(ReduceSum, opset_version, attrs, inputs)


def test_reducesum_11():
    opset_version = 13
    keepdims = 1
    noop_with_empty_axes = 0
    attrs = {"keepdims": keepdims, "noop_with_empty_axes": noop_with_empty_axes}

    x = np.random.randn(3, 2, 2).astype(np.float32)
    axes = np.array([], dtype=np.int64)
    inputs = [x, axes]

    check(ReduceSum, opset_version, attrs, inputs)
