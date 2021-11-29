import numpy as np
from onnion_runtime import Concat

from .utils import check


def test_concat_00():
    opset_version = 13
    axis = 1
    attrs = {"axis": axis}

    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)
    v2 = np.random.randn(2, 3).astype(np.float32)
    inputs = [v0, v1, v2]

    check(Concat, opset_version, attrs, inputs)
