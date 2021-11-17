import numpy as np
from onnion_runtime import Acosh

from .utils import check


def test_acosh_00():
    opset_version = 13
    attrs = dict()

    x = np.random.uniform(1.0, 10.0, (3, 4, 5)).astype(np.float32)
    inputs = [x]

    check(Acosh, opset_version, attrs, inputs)
