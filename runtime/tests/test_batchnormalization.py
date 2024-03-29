import numpy as np
from onnion_runtime import BatchNormalization

from .utils import check


def test_batchnormalization_00():
    opset_version = 9
    attrs = dict()

    # input size: (2, 3, 4, 5)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)

    inputs = [x, s, bias, mean, var]
    check(BatchNormalization, opset_version, attrs, inputs)


def test_batchnormalization_01():
    opset_version = 9
    attrs = dict()

    # input size: (2, 3, 4, 5)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    epsilon = 1e-2

    inputs = [x, s, bias, mean, var]
    attrs["epsilon"] = epsilon
    check(BatchNormalization, opset_version, attrs, inputs)


def test_batchnormalization_02():
    opset_version = 14
    attrs = dict()

    # input size: (2, 3, 4, 5)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)

    inputs = [x, s, bias, mean, var]
    check(BatchNormalization, opset_version, attrs, inputs)


def test_batchnormalization_03():
    opset_version = 15
    attrs = dict()

    # input size: (2, 3, 4, 5)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)

    inputs = [x, s, bias, mean, var]
    check(BatchNormalization, opset_version, attrs, inputs)
