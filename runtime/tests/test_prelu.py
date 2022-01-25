import numpy as np
from onnion_runtime import PRelu

from .utils import check


def test_prelu_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x, slope]

    check(PRelu, opset_version, attrs, inputs)


def test_prelu_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(5).astype(np.float32)
    inputs = [x, slope]

    check(PRelu, opset_version, attrs, inputs)
