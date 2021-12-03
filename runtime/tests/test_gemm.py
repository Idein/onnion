import numpy as np
from onnion_runtime import Gemm

from .utils import check


def test_gemm_00():
    opset_version = 13
    alpha = 0.25
    beta = 0.35
    transA = 1
    transB = 1
    attrs = {"alpha": alpha, "beta": beta, "transA": transA, "transB": transB}

    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_01():
    opset_version = 13
    alpha = 0.5
    attrs = {"alpha": alpha}

    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_02():
    opset_version = 13
    beta = 0.5
    attrs = {"beta": beta}

    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_03():
    opset_version = 13
    attrs = dict()

    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.random.ranf([3, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_04():
    opset_version = 13
    attrs = dict()

    a = np.random.ranf([2, 10]).astype(np.float32)
    b = np.random.ranf([10, 3]).astype(np.float32)
    inputs = [a, b]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_05():
    opset_version = 13
    attrs = dict()

    a = np.random.ranf([2, 3]).astype(np.float32)
    b = np.random.ranf([3, 4]).astype(np.float32)
    c = np.array(3.14).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_06():
    opset_version = 13
    attrs = dict()

    a = np.random.ranf([3, 7]).astype(np.float32)
    b = np.random.ranf([7, 3]).astype(np.float32)
    c = np.random.ranf([1]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_07():
    opset_version = 13
    attrs = dict()

    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_08():
    opset_version = 13
    attrs = dict()

    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_09():
    opset_version = 13
    transA = 1
    attrs = {"transA": transA}

    a = np.random.ranf([6, 3]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)


def test_gemm_10():
    opset_version = 13
    transB = 1
    attrs = {"transB": transB}

    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([4, 6]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    inputs = [a, b, c]

    check(Gemm, opset_version, attrs, inputs)
