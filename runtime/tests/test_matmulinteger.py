import numpy as np
from onnion_runtime import MatMulInteger

from .utils import check


def test_matmulinteger_00():
    opset_version = 13
    attrs = dict()

    x = np.array(
        [
            [11, 7, 3],
            [10, 6, 2],
            [9, 5, 1],
            [8, 4, 0],
        ],
        dtype=np.uint8,
    )
    y = np.array(
        [
            [1, 4],
            [2, 5],
            [3, 6],
        ],
        dtype=np.uint8,
    )
    inputs = [x, y]

    check(MatMulInteger, opset_version, attrs, inputs)


def test_matmulinteger_01():
    opset_version = 13
    attrs = dict()

    x = np.array(
        [
            [11, 7, 3],
            [10, 6, 2],
            [9, 5, 1],
            [8, 4, 0],
        ],
        dtype=np.uint8,
    )
    x_zero_point = np.array([12], dtype=np.uint8)
    y = np.array(
        [
            [1, 4],
            [2, 5],
            [3, 6],
        ],
        dtype=np.uint8,
    )
    inputs = [x, y, x_zero_point]

    check(MatMulInteger, opset_version, attrs, inputs)


def test_matmulinteger_02():
    opset_version = 13
    attrs = dict()

    x = np.array(
        [
            [11, 7, 3],
            [10, 6, 2],
            [9, 5, 1],
            [8, 4, 0],
        ],
        dtype=np.uint8,
    )
    x_zero_point = np.array([12], dtype=np.uint8)
    y = np.array(
        [
            [1, 4],
            [2, 5],
            [3, 6],
        ],
        dtype=np.uint8,
    )
    y_zero_point = np.array([4], dtype=np.uint8)
    inputs = [x, y, x_zero_point, y_zero_point]

    check(MatMulInteger, opset_version, attrs, inputs)
