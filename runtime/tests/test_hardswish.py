import numpy as np
from onnion_runtime import HardSwish

from .utils import check


def test_hardswish_00():
    opset_version = 14
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(HardSwish, opset_version, attrs, inputs)
