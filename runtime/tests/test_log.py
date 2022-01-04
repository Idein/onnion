import numpy as np
from onnion_runtime import Log

from .utils import check


def test_log_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(1, 2, 3, 4).astype(np.float32)
    inputs = [np.exp(x)]

    check(Log, opset_version, attrs, inputs)
