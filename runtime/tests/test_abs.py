import numpy as np
from onnion_runtime import Abs

from .utils import check


def test_abs_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Abs(opset_version).run(x)

    check("Abs", dict(), [x], outputs, opset_version)
