import numpy as np
from onnion_runtime import Squeeze

from .utils import check


def test_squeeze_00():
    opset_version = 11
    axes = [0]
    attrs = {"axes": axes}

    data = np.random.randn(1, 3, 4, 5).astype(np.float32)
    inputs = [data]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_01():
    opset_version = 11
    axes = [-2]
    attrs = {"axes": axes}

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    inputs = [data]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_02():
    opset_version = 11
    axes = [0, 2]
    attrs = {"axes": axes}

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    inputs = [data]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_03():
    opset_version = 11
    attrs = dict()

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    inputs = [data]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_04():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(1, 3, 4, 5).astype(np.float32)
    axes = np.array([0]).astype(np.int64)
    inputs = [data, axes]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_05():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    inputs = [data, axes]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_06():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([0, 2]).astype(np.int64)
    inputs = [data, axes]

    check(Squeeze, opset_version, attrs, inputs)


def test_squeeze_07():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(1, 3, 1, 5).astype(np.float32)
    inputs = [data]

    check(Squeeze, opset_version, attrs, inputs)
