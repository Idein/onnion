import numpy as np
from onnion_runtime import Exp

from .utils import check


def test_exp_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 2, 3, 4).astype(np.float32)
    inputs = [x]

    check(Exp, opset_version, attrs, inputs)
