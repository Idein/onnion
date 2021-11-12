import numpy as np
from onnion_runtime import Reshape

from .utils import check


def test_reshape_00():
    opset_version = 13

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([4, 2, 3]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_01():
    opset_version = 14
    allowzero = 1

    data = np.random.randn(0, 3, 4).astype(np.float32)
    shape = np.array([3, 4, 0]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version, allowzero=allowzero).run(data, shape)
    check("Reshape", {"allowzero": allowzero}, inputs, outputs, opset_version)


def test_reshape_02():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([4, 2, 3]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_03():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, 4, 3]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_04():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, 12]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_05():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, 3, 2, 2]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_06():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([24]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_07():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, -1, 2]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_08():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([-1, 2, 3, 4]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_09():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, 0, 4, 1]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)


def test_reshape_10():
    opset_version = 14

    data = np.random.randn(2, 3, 4).astype(np.float32)
    shape = np.array([2, 0, 1, -1]).astype(np.int64)
    inputs = [data, shape]
    outputs = Reshape(opset_version).run(data, shape)
    check("Reshape", dict(), inputs, outputs, opset_version)
