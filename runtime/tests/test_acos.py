import numpy as np
from onnion_runtime import Acos

from .utils import check


def test_acos_00():
    opset_version = 13
    attrs = dict()

    x = np.random.rand(3, 4, 5).astype(np.float32)
    inputs = [x]
    check(Acos, opset_version, attrs, inputs)
