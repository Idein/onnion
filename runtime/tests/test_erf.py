import numpy as np
from onnion_runtime import Erf

from .utils import check


def test_erf_00():
    opset_version = 13

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    outputs = Erf(opset_version).run(x)

    check("Erf", dict(), [x], outputs, opset_version)
