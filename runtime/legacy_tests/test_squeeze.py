import numpy as np
from onnion_runtime import Squeeze

from .utils import check


def test_squeeze_00():
    opset_version = 11
    axes = [0]

    data = np.random.randn(1, 3, 4, 5).astype(np.float32)
    outputs = Squeeze(opset_version, axes=axes).run(data)
    check("Squeeze", {"axes": axes}, [data], outputs, opset_version)


def test_squeeze_01():
    opset_version = 11
    axes = [-2]

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    outputs = Squeeze(opset_version, axes=axes).run(data)
    check("Squeeze", {"axes": axes}, [data], outputs, opset_version)


def test_squeeze_02():
    opset_version = 11
    axes = [0, 2]

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    outputs = Squeeze(opset_version, axes=axes).run(data)
    check("Squeeze", {"axes": axes}, [data], outputs, opset_version)


def test_squeeze_03():
    opset_version = 11

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    outputs = Squeeze(opset_version).run(data)
    check("Squeeze", dict(), [data], outputs, opset_version)


def test_squeeze_04():
    opset_version = 13

    data = np.random.randn(1, 3, 4, 5).astype(np.float32)
    axes = np.array([0]).astype(np.int64)
    inputs = [data, axes]
    outputs = Squeeze(opset_version).run(data, axes)
    check("Squeeze", dict(), inputs, outputs, opset_version)


def test_squeeze_05():
    opset_version = 13

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    inputs = [data, axes]
    outputs = Squeeze(opset_version).run(data, axes)
    check("Squeeze", dict(), inputs, outputs, opset_version)


def test_squeeze_06():
    opset_version = 13

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([0, 2]).astype(np.int64)
    inputs = [data, axes]
    outputs = Squeeze(opset_version).run(data, axes)
    check("Squeeze", dict(), inputs, outputs, opset_version)


def test_squeeze_07():
    opset_version = 13

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    inputs = [data]
    outputs = Squeeze(opset_version).run(data)
    check("Squeeze", dict(), inputs, outputs, opset_version)
