import numpy as np
from onnion_runtime import Unsqueeze

from .utils import check


def test_unsqueeze_00():
    opset_version = 11
    axes = [1]
    attrs = {"axes": axes}

    data = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [data]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_01():
    opset_version = 11
    axes = [-1]
    attrs = {"axes": axes}

    data = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [data]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_02():
    opset_version = 11
    axes = [1, 4]
    attrs = {"axes": axes}

    data = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [data]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_03():
    opset_version = 11
    axes = [5, 4, 2]
    attrs = {"axes": axes}

    data = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [data]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_04():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1]).astype(np.int64)
    inputs = [data, axes]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_05():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([-1]).astype(np.int64)
    inputs = [data, axes]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_06():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1, 4]).astype(np.int64)
    inputs = [data, axes]

    check(Unsqueeze, opset_version, attrs, inputs)


def test_unsqueeze_07():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)
    inputs = [data, axes]

    check(Unsqueeze, opset_version, attrs, inputs)
