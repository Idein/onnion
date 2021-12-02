import numpy as np
from onnion_runtime import Det

from .utils import check


def test_det_00():
    opset_version = 13
    attrs = dict()

    x = np.arange(4).reshape(2, 2).astype(np.float32)
    inputs = [x]

    check(Det, opset_version, attrs, inputs)


def test_det_01():
    opset_version = 13
    attrs = dict()

    x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)
    inputs = [x]

    check(Det, opset_version, attrs, inputs)
