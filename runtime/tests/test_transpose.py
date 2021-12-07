import numpy as np
from onnion_runtime import Transpose

from .utils import check


def test_transpose_00():
    opset_version = 13
    attrs = dict()

    data = np.random.randn(2, 3, 4).astype(np.float32)
    inputs = [data]

    check(Transpose, opset_version, attrs, inputs)


def test_transpose_01():
    opset_version = 13
    perm = [1, 0, 2]
    attrs = {"perm": perm}

    data = np.random.randn(2, 3, 4).astype(np.float32)
    inputs = [data]

    check(Transpose, opset_version, attrs, inputs)
