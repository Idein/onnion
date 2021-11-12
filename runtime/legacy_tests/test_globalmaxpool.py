import numpy as np
from onnion_runtime import GlobalMaxPool

from .utils import check


def test_globalmaxpool_00():
    opset_version = 13

    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    outputs = GlobalMaxPool(opset_version).run(x)

    check("GlobalMaxPool", dict(), [x], outputs, opset_version)


def test_globalmaxpool_01():
    opset_version = 13

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
    outputs = GlobalMaxPool(opset_version).run(x)

    check("GlobalMaxPool", dict(), [x], outputs, opset_version)
