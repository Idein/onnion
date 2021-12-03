import numpy as np
from onnion_runtime import GlobalMaxPool

from .utils import check


def test_globalmaxpool_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    inputs = [x]

    check(GlobalMaxPool, opset_version, attrs, inputs)


def test_globalmaxpool_01():
    opset_version = 13
    attrs = dict()

    x = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    ).astype(np.float32)
    inputs = [x]

    check(GlobalMaxPool, opset_version, attrs, inputs)
