import numpy as np
from onnion_runtime import HardSwish

from .utils import check


def test_hardswish_00():
    opset_version = 14

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = HardSwish(opset_version).run(x)

    check("HardSwish", dict(), [x], outputs, opset_version)
