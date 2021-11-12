import numpy as np
from onnion_runtime import Flatten

from .utils import check


def test_flatten_00():
    opset_version = 13

    axis = 3
    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    outputs = Flatten(opset_version, axis=axis).run(x)

    check("Flatten", {"axis": axis}, [x], outputs, opset_version)


def test_flatten_01():
    opset_version = 13

    axis = 0
    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    outputs = Flatten(opset_version, axis=axis).run(x)

    check("Flatten", {"axis": axis}, [x], outputs, opset_version)


def test_flatten_02():
    opset_version = 13

    axis = -2
    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    outputs = Flatten(opset_version, axis=axis).run(x)

    check("Flatten", {"axis": axis}, [x], outputs, opset_version)


def test_flatten_03():
    opset_version = 13

    shape = (2, 3, 4, 5)
    x = np.random.random_sample(shape).astype(np.float32)
    outputs = Flatten(opset_version).run(x)

    check("Flatten", dict(), [x], outputs, opset_version)
