import numpy as np
from onnion_runtime import Asin

from .utils import check


def test_asin_00():
    opset_version = 13
    attrs = dict()

    x = np.random.rand(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Asin, opset_version, attrs, inputs)
