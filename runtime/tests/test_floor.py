import numpy as np
from onnion_runtime import Floor

from .utils import check


def test_floor_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Floor(opset_version).run(x)

    check("Floor", dict(), [x], outputs, opset_version)
