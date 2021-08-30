import numpy as np
from onnion_runtime import Compress

from .utils import check


def test_compress_00():
    opset_version = 13

    axis = 0
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    condition = np.array([0, 1, 1]).astype(bool)
    inputs = [x, condition]
    outputs = Compress(opset_version, axis=axis).run(x, condition)

    check("Compress", {"axis": axis}, inputs, outputs, opset_version)


def test_compress_01():
    opset_version = 13

    axis = 1
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    condition = np.array([0, 1]).astype(bool)
    inputs = [x, condition]
    outputs = Compress(opset_version, axis=axis).run(x, condition)

    check("Compress", {"axis": axis}, inputs, outputs, opset_version)


def test_compress_02():
    opset_version = 13

    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    condition = np.array([0, 1, 0, 0, 1]).astype(bool)
    inputs = [x, condition]
    outputs = Compress(opset_version).run(x, condition)

    check("Compress", dict(), inputs, outputs, opset_version)


def test_compress_03():
    opset_version = 13

    axis = -1
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    condition = np.array([0, 1]).astype(bool)
    inputs = [x, condition]
    outputs = Compress(opset_version, axis=axis).run(x, condition)

    check("Compress", {"axis": axis}, inputs, outputs, opset_version)
