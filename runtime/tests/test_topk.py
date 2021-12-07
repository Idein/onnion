import numpy as np
from onnion_runtime import TopK

from .utils import check


def test_topk_00():
    opset_version = 1
    axis = 1
    k = 3
    attrs = {"axis": axis, "k": k}

    x = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.float32,
    )
    inputs = [x]

    check(TopK, opset_version, attrs, inputs)


def test_topk_01():
    opset_version = 10
    axis = 1
    attrs = {"axis": axis}

    x = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.float32,
    )
    k = np.array([3]).astype(np.int64)
    inputs = [x, k]

    check(TopK, opset_version, attrs, inputs)


def test_topk_02():
    opset_version = 11
    axis = 1
    attrs = {"axis": axis}

    x = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.float32,
    )
    k = np.array([3]).astype(np.int64)
    inputs = [x, k]

    check(TopK, opset_version, attrs, inputs)


def test_topk_03():
    opset_version = 11
    axis = -1
    attrs = {"axis": axis}

    x = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.float32,
    )
    k = np.array([3]).astype(np.int64)
    inputs = [x, k]

    check(TopK, opset_version, attrs, inputs)


def test_topk_04():
    opset_version = 11
    axis = 1
    largest = 0
    sorted = 1
    attrs = {"axis": axis, "largest": largest, "sorted": sorted}

    x = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.float32,
    )
    k = np.array([3]).astype(np.int64)
    inputs = [x, k]

    check(TopK, opset_version, attrs, inputs)
