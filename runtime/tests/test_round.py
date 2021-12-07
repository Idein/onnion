import numpy as np
from onnion_runtime import Round

from .utils import check


def test_round_00():
    opset_version = 13
    attrs = dict()

    x = np.array([0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2, -2.5, -2.8]).astype(np.float32)
    inputs = [x]

    check(Round, opset_version, attrs, inputs)
