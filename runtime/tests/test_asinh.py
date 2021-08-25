import numpy as np
from onnion_runtime import Asinh

from .utils import check


def test_asinh_00():
    opset_version = 13

    x = np.random.rand(3, 4, 5).astype(np.float32)
    outputs = Asinh(opset_version).run(x)

    check("Asinh", dict(), [x], outputs, opset_version)
