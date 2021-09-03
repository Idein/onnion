import numpy as np
from onnion_runtime import Dropout

from .utils import check


def test_dropout_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Dropout(opset_version).run(x)

    check("Dropout", dict(), [x], outputs, opset_version)


def test_dropout_01():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.array(0.1).astype(np.float32)
    inputs = [x, r]
    outputs = Dropout(opset_version).run(x, r)

    check("Dropout", dict(), inputs, outputs, opset_version)


def test_dropout_02():
    opset_version = 11

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Dropout(opset_version).run(x)

    check("Dropout", dict(), [x], outputs, opset_version)
