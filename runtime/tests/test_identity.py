import numpy as np
from onnion_runtime import Identity

from .utils import check


def test_identity_00():
    opset_version = 14
    attrs = dict()

    data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    inputs = [data]

    check(Identity, opset_version, attrs, inputs)
