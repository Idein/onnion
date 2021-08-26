import numpy as np
from onnion_runtime import Atan

from .utils import check


def test_atan_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Atan(opset_version).run(x)

    check("Atan", dict(), [x], outputs, opset_version)
