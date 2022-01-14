import numpy as np
from onnion_runtime import MatMul

from .utils import check


def test_matmul_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(4, 3).astype(np.float32)
    inputs = [x, y]

    check(MatMul, opset_version, attrs, inputs)


def test_matmul_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 4, 3).astype(np.float32)
    inputs = [x, y]

    check(MatMul, opset_version, attrs, inputs)


def test_matmul_02():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 2, 3, 4).astype(np.float32)
    y = np.random.randn(1, 2, 4, 3).astype(np.float32)
    inputs = [x, y]

    check(MatMul, opset_version, attrs, inputs)
