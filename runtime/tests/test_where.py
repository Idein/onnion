import numpy as np
from onnion_runtime import Where

from .utils import check


def test_where_00():
    opset_version = 13
    attrs = dict()

    condition = np.array([[1, 0], [1, 1]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[9, 8], [7, 6]], dtype=np.int64)
    inputs = [condition, x, y]

    check(Where, opset_version, attrs, inputs)


def test_where_01():
    opset_version = 13
    attrs = dict()

    condition = np.array([[1, 0], [1, 1]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)
    inputs = [condition, x, y]

    check(Where, opset_version, attrs, inputs)
