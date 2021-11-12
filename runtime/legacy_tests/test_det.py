import numpy as np
from onnion_runtime import Det

from .utils import check


def test_det_00():
    opset_version = 13

    x = np.arange(4).reshape(2, 2).astype(np.float32)
    outputs = Det(opset_version).run(x)

    check("Det", dict(), [x], outputs, opset_version)


def test_det_01():
    opset_version = 13

    x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)
    outputs = Det(opset_version).run(x)

    check("Det", dict(), [x], outputs, opset_version)
