import numpy as np
from onnion_runtime import Elu

from .utils import check


def test_elu_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Elu, opset_version, attrs, inputs)


def test_elu_01():
    opset_version = 13
    alpha = 2.0
    attrs = {"alpha": alpha}

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Elu, opset_version, attrs, inputs)
