import numpy as np
from onnion_runtime import Softmax

from .utils import check


def test_softmax_00():
    opset_version = 13
    attrs = {"axis": 1}

    x = np.array([[-1, 0, 1]]).astype(np.float32)
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_01():
    opset_version = 13
    attrs = dict()

    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_02():
    opset_version = 13
    attrs = {"axis": 0}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_03():
    opset_version = 13
    attrs = {"axis": 1}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_04():
    opset_version = 13
    attrs = {"axis": 2}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_05():
    opset_version = 13
    attrs = {"axis": -1}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_06():
    opset_version = 13
    attrs = dict()

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_07():
    opset_version = 11
    attrs = {"axis": 0}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_08():
    opset_version = 11
    attrs = {"axis": 1}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_09():
    opset_version = 11
    attrs = {"axis": -1}

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)


def test_softmax_10():
    opset_version = 11
    attrs = dict()

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    inputs = [x]

    check(Softmax, opset_version, attrs, inputs)
