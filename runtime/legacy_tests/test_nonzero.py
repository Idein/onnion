import numpy as np
from onnion_runtime import NonZero

from .utils import check


def test_nonzero_00():
    opset_version = 13

    x = np.array([[1, 0], [1, 1]], dtype=bool)
    outputs = NonZero(opset_version).run(x)

    check("NonZero", dict(), [x], outputs, opset_version)
