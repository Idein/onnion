import numpy as np
from onnion_runtime import Neg

from .utils import check


def test_neg_00():
    opset_version = 13
    attrs = dict()

    x = np.array([-4, 2]).astype(np.float32)
    inputs = [x]

    check(Neg, opset_version, attrs, inputs)


def test_neg_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Neg, opset_version, attrs, inputs)
