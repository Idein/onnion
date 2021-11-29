import numpy as np
from onnion_runtime import Celu

from .utils import check


def test_celu_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Celu, opset_version, attrs, inputs)


def test_celu_01():
    opset_version = 13
    alpha = 2.0
    attrs = {"alpha": alpha}

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Celu, opset_version, attrs, inputs)
