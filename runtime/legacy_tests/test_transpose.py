import numpy as np
from onnion_runtime import Transpose

from .utils import check


def test_transpose_00():
    opset_version = 13
    data = np.random.randn(2, 3, 4).astype(np.float32)

    outputs = Transpose(opset_version).run(data)
    check("Transpose", dict(), [data], outputs, opset_version)


def test_transpose_01():
    opset_version = 13
    data = np.random.randn(2, 3, 4).astype(np.float32)
    perm = [1, 0, 2]

    outputs = Transpose(opset_version, perm=perm).run(data)
    check("Transpose", {"perm": perm}, [data], outputs, opset_version)
