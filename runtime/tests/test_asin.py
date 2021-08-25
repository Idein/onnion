import numpy as np
from onnion_runtime import Asin

from .utils import check


def test_asin_00():
    opset_version = 13

    x = np.random.rand(3, 4, 5).astype(np.float32)
    outputs = Asin(opset_version).run(x)

    check("Asin", dict(), [x], outputs, opset_version)
