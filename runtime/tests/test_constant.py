import numpy as np
from onnion_runtime import Constant

from .utils import check


def test_constant_00():
    opset_version = 14
    v = np.array([1, 2, 3, 4]).astype(np.float32)
    attrs = {"value": v}

    inputs = []

    check(Constant, opset_version, attrs, inputs)
