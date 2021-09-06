import numpy as np
from onnion_runtime import Einsum

from .utils import check


def test_einsum_00():
    opset_version = 13

    eqn = "...ii ->...i"
    x = np.random.randn(3, 5, 5).astype(np.float32)
    inputs = [x]
    outputs = Einsum(opset_version, equation=eqn).run(x)

    check("Einsum", {"equation": eqn}, inputs, outputs, opset_version)


def test_einsum_01():
    opset_version = 13

    eqn = "bij, bjk -> bik"
    x = np.random.randn(5, 2, 3).astype(np.float32)
    y = np.random.randn(5, 3, 4).astype(np.float32)
    inputs = [x, y]
    outputs = Einsum(opset_version, equation=eqn).run(x, y)

    check("Einsum", {"equation": eqn}, inputs, outputs, opset_version)


def test_einsum_02():
    opset_version = 13

    eqn = "i,i"
    x = np.random.randn(5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    inputs = [x, y]
    outputs = Einsum(opset_version, equation=eqn).run(x, y)

    check("Einsum", {"equation": eqn}, inputs, outputs, opset_version)


def test_einsum_03():
    opset_version = 13

    eqn = "ij->i"
    x = np.random.randn(3, 4).astype(np.float32)
    inputs = [x]
    outputs = Einsum(opset_version, equation=eqn).run(x)

    check("Einsum", {"equation": eqn}, inputs, outputs, opset_version)


def test_einsum_04():
    opset_version = 13

    eqn = "ij->ji"
    x = np.random.randn(3, 4).astype(np.float32)
    inputs = [x]
    outputs = Einsum(opset_version, equation=eqn).run(x)

    check("Einsum", {"equation": eqn}, inputs, outputs, opset_version)
