import numpy as np
from onnion_runtime import Ceil

from .utils import check


def test_ceil_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Ceil(opset_version).run(x)

    check("Ceil", dict(), [x], outputs, opset_version)
