import numpy as np
from onnion_runtime import Acos

from .utils import check


def test_acos_00():
    opset_version = 13

    x = np.random.rand(3, 4, 5).astype(np.float32)
    outputs = Acos(opset_version).run(x)

    check("Acos", dict(), [x], outputs, opset_version)
