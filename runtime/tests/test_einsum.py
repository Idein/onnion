import numpy as np
from onnion_runtime import Einsum

from .utils import check


def test_einsum_00():
    opset_version = 13
    eqn = "...ii ->...i"
    attrs = {"equation": eqn}

    x = np.random.randn(3, 5, 5).astype(np.float32)
    inputs = [x]

    check(Einsum, opset_version, attrs, inputs)


def test_einsum_01():
    opset_version = 13
    eqn = "bij, bjk -> bik"
    attrs = {"equation": eqn}

    x = np.random.randn(5, 2, 3).astype(np.float32)
    y = np.random.randn(5, 3, 4).astype(np.float32)
    inputs = [x, y]

    check(Einsum, opset_version, attrs, inputs)


def test_einsum_02():
    opset_version = 13
    eqn = "i,i"
    attrs = {"equation": eqn}

    x = np.random.randn(5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    inputs = [x, y]

    check(Einsum, opset_version, attrs, inputs)


def test_einsum_03():
    opset_version = 13
    eqn = "ij->i"
    attrs = {"equation": eqn}

    x = np.random.randn(3, 4).astype(np.float32)
    inputs = [x]

    check(Einsum, opset_version, attrs, inputs)


def test_einsum_04():
    opset_version = 13
    eqn = "ij->ji"
    attrs = {"equation": eqn}

    x = np.random.randn(3, 4).astype(np.float32)
    inputs = [x]

    check(Einsum, opset_version, attrs, inputs)
