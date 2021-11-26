import numpy as np
from onnion_runtime import Cosh

from .utils import check


def test_cosh_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Cosh, opset_version, attrs, inputs)
