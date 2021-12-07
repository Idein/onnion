import numpy as np
from onnion_runtime import NonZero

from .utils import check


def test_nonzero_00():
    opset_version = 13
    attrs = dict()

    x = np.array([[1, 0], [1, 1]], dtype=bool)
    inputs = [x]

    check(NonZero, opset_version, attrs, inputs)
