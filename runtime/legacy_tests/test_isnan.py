import numpy as np
from onnion_runtime import IsNaN

from .utils import check


def test_isnan_00():
    opset_version = 13

    x = np.array([3.0, np.nan, 4.0, np.nan], dtype=np.float32)
    outputs = IsNaN(opset_version).run(x)

    check("IsNaN", dict(), [x], outputs, opset_version)
