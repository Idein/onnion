import numpy as np
from onnion_runtime import Pow

from .utils import check


def test_pow_00():
    opset_version = 15
    attrs = dict()

    np.random.seed(0)
    v0 = np.arange(6).reshape(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)
    inputs = [v0, v1]

    check(Pow, opset_version, attrs, inputs)


def test_pow_01():
    opset_version = 15
    attrs = dict()

    np.random.seed(0)
    v0 = np.arange(6).reshape(2, 3).astype(np.float32)
    v1 = np.random.randn(3).astype(np.float32)
    inputs = [v0, v1]

    check(Pow, opset_version, attrs, inputs)


def test_pow_02():
    opset_version = 15
    attrs = dict()

    v0 = np.array([1, 2, 3]).astype(np.float32)
    v1 = np.array([4, 5, 6]).astype(np.int64)
    inputs = [v0, v1]

    check(Pow, opset_version, attrs, inputs)


def test_pow_03():
    opset_version = 15
    attrs = dict()

    v0 = np.array([1, 2, 3]).astype(np.int64)
    v1 = np.array([4, 5, 6]).astype(np.float32)
    inputs = [v0, v1]

    check(Pow, opset_version, attrs, inputs)


def test_pow_04():
    opset_version = 15
    attrs = dict()

    v0 = np.array([1, 2, 3]).astype(np.int64)
    v1 = np.array([4, 5, 6]).astype(np.int64)
    inputs = [v0, v1]

    check(Pow, opset_version, attrs, inputs)
