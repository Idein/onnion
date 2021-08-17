import numpy as np
from onnion_runtime import Identity

from .utils import check


def test_identity_00():
    opset_version = 14

    data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    outputs = Identity(opset_version).run(data)

    check("Identity", dict(), [data], outputs, opset_version)
