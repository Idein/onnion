import numpy as np
from onnion_runtime import TopK

from .utils import check


def test_topk_00():
    opset_version = 1
    axis = 1
    k = 3

    x = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.float32,
    )
    outputs = TopK(opset_version, k=k, axis=axis).run(x)
    check("TopK", {"axis": axis, "k": k}, [x], outputs, opset_version)


def test_topk_01():
    opset_version = 10
    axis = 1

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
    outputs = TopK(opset_version, axis=axis).run(x, k)
    check("TopK", {"axis": axis}, inputs, outputs, opset_version)


def test_topk_02():
    opset_version = 11
    axis = 1

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
    outputs = TopK(opset_version, axis=axis).run(x, k)
    check("TopK", {"axis": axis}, inputs, outputs, opset_version)


def test_topk_03():
    opset_version = 11
    axis = -1

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
    outputs = TopK(opset_version, axis=axis).run(x, k)
    check("TopK", {"axis": axis}, inputs, outputs, opset_version)


def test_topk_04():
    opset_version = 11
    axis = 1
    largest = 0
    sorted = 1

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
    outputs = TopK(opset_version, axis=axis, largest=largest, sorted=sorted).run(x, k)
    check("TopK", {"axis": axis, "largest": largest, "sorted": sorted}, inputs, outputs, opset_version)
