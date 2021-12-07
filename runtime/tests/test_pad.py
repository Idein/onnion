import numpy as np
from onnion_runtime import Pad

from .utils import check


def test_pad_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
    value = np.array(1.2).astype(np.float32)
    inputs = [x, pads, value]

    check(Pad, opset_version, attrs, inputs)


def test_pad_01():
    opset_version = 13
    mode = "edge"
    attrs = {"mode": mode}

    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
    inputs = [x, pads]

    check(Pad, opset_version, attrs, inputs)


def test_pad_02():
    opset_version = 13
    mode = "reflect"
    attrs = {"mode": mode}

    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
    inputs = [x, pads]

    check(Pad, opset_version, attrs, inputs)
