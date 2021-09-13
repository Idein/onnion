import numpy as np
from onnion_runtime import Pad

from .utils import check


def test_pad_00():
    opset_version = 13

    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
    value = np.array(1.2).astype(np.float32)
    inputs = [x, pads, value]
    outputs = Pad(opset_version).run(x, pads, value)

    check("Pad", dict(), inputs, outputs, opset_version)


def test_pad_01():
    opset_version = 13

    mode = "edge"
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
    inputs = [x, pads]
    outputs = Pad(opset_version, mode=mode).run(x, pads)

    check("Pad", {"mode": mode}, inputs, outputs, opset_version)


def test_pad_02():
    opset_version = 13

    mode = "reflect"
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
    inputs = [x, pads]
    outputs = Pad(opset_version, mode=mode).run(x, pads)

    check("Pad", {"mode": mode}, inputs, outputs, opset_version)
