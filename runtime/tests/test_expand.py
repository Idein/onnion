import numpy as np
from onnion_runtime import Expand

from .utils import check


def test_expand_00():
    opset_version = 13

    shape = np.array([3, 1]).astype(np.int64)
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    new_shape = np.array([2, 1, 6]).astype(np.int64)
    inputs = [data, new_shape]
    outputs = Expand(opset_version).run(data, new_shape)

    check("Expand", dict(), inputs, outputs, opset_version)


def test_expand_01():
    opset_version = 13

    shape = np.array([3, 1]).astype(np.int64)
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    new_shape = np.array([3, 4]).astype(np.int64)
    inputs = [data, new_shape]
    outputs = Expand(opset_version).run(data, new_shape)

    check("Expand", dict(), inputs, outputs, opset_version)
