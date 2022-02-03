import numpy as np
from onnion_runtime import Relu

from .utils import check


def test_relu_00():
    opset_version = 14
    attrs = dict()

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(Relu, opset_version, attrs, inputs)
