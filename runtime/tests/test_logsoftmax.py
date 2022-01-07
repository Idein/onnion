import numpy as np
from onnion_runtime import LogSoftmax

from .utils import check


def test_logsoftmax_00():
    opset_version = 13
    attrs = dict()

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(LogSoftmax, opset_version, attrs, inputs)


def test_logsoftmax_01():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(LogSoftmax, opset_version, attrs, inputs)


def test_logsoftmax_02():
    opset_version = 13
    axis = 1
    attrs = {"axis": axis}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(LogSoftmax, opset_version, attrs, inputs)


def test_logsoftmax_03():
    opset_version = 11
    attrs = dict()

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(LogSoftmax, opset_version, attrs, inputs)


def test_logsoftmax_04():
    opset_version = 11
    axis = 0
    attrs = {"axis": axis}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(LogSoftmax, opset_version, attrs, inputs)


def test_logsoftmax_05():
    opset_version = 11
    axis = -1
    attrs = {"axis": axis}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(LogSoftmax, opset_version, attrs, inputs)
