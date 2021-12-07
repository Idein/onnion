import numpy as np
from onnion_runtime import Shape

from .utils import check


def test_shape_00():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Shape, opset_version, attrs, inputs)
