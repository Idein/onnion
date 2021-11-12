import numpy as np
from onnion_runtime import Unsqueeze

from .utils import check


def test_unsqueeze_00():
    opset_version = 11
    axes = [1]

    data = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Unsqueeze(opset_version, axes=axes).run(data)
    check("Unsqueeze", {"axes": axes}, [data], outputs, opset_version)


def test_unsqueeze_01():
    opset_version = 11
    axes = [-1]

    data = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Unsqueeze(opset_version, axes=axes).run(data)
    check("Unsqueeze", {"axes": axes}, [data], outputs, opset_version)


def test_unsqueeze_02():
    opset_version = 11
    axes = [1, 4]

    data = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Unsqueeze(opset_version, axes=axes).run(data)
    check("Unsqueeze", {"axes": axes}, [data], outputs, opset_version)


def test_unsqueeze_03():
    opset_version = 11
    axes = [5, 4, 2]

    data = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Unsqueeze(opset_version, axes=axes).run(data)
    check("Unsqueeze", {"axes": axes}, [data], outputs, opset_version)


def test_unsqueeze_04():
    opset_version = 13

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1]).astype(np.int64)
    inputs = [data, axes]
    outputs = Unsqueeze(opset_version).run(data, axes)
    check("Unsqueeze", dict(), inputs, outputs, opset_version)


def test_unsqueeze_05():
    opset_version = 13

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([-1]).astype(np.int64)
    inputs = [data, axes]
    outputs = Unsqueeze(opset_version).run(data, axes)
    check("Unsqueeze", dict(), inputs, outputs, opset_version)


def test_unsqueeze_06():
    opset_version = 13

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1, 4]).astype(np.int64)
    inputs = [data, axes]
    outputs = Unsqueeze(opset_version).run(data, axes)
    check("Unsqueeze", dict(), inputs, outputs, opset_version)


def test_unsqueeze_07():
    opset_version = 13

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)
    inputs = [data, axes]
    outputs = Unsqueeze(opset_version).run(data, axes)
    check("Unsqueeze", dict(), inputs, outputs, opset_version)
