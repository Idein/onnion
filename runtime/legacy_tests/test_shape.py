import numpy as np
from onnion_runtime import Shape

from .utils import check


def test_shape_00():
    opset_version = 13

    v0 = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Shape(opset_version).run(v0)
    check("Shape", dict(), [v0], outputs, opset_version)
