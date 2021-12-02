import numpy as np
from onnion_runtime import Erf

from .utils import check


def test_erf_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    inputs = [x]

    check(Erf, opset_version, attrs, inputs)
