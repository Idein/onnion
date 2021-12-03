import numpy as np
from onnion_runtime import Flatten

from .utils import check


def test_flatten_00():
    opset_version = 13
    axis = 3
    attrs = {"axis": axis}

    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    inputs = [x]

    check(Flatten, opset_version, attrs, inputs)


def test_flatten_01():
    opset_version = 13
    axis = 0
    attrs = {"axis": axis}

    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    inputs = [x]

    check(Flatten, opset_version, attrs, inputs)


def test_flatten_02():
    opset_version = 13
    axis = -2
    attrs = {"axis": axis}

    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    inputs = [x]

    check(Flatten, opset_version, attrs, inputs)


def test_flatten_03():
    opset_version = 13
    attrs = dict()

    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    inputs = [x]

    check(Flatten, opset_version, attrs, inputs)
