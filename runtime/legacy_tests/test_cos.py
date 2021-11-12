import numpy as np
from onnion_runtime import Cos

from .utils import check


def test_cos_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Cos(opset_version).run(x)

    check("Cos", dict(), [x], outputs, opset_version)
