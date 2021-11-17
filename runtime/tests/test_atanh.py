import numpy as np
from onnion_runtime import Atanh

from .utils import check


def test_atanh_00():
    opset_version = 13
    attrs = dict()

    x = np.random.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float32)
    inputs = [x]

    check(Atanh, opset_version, attrs, inputs)
