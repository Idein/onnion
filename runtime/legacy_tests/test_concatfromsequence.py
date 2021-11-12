import numpy as np
from onnion_runtime import ConcatFromSequence

from .utils import check


def test_concatfromsequence_00():
    opset_version = 13
    axis = 1
    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)
    v2 = np.random.randn(2, 3).astype(np.float32)

    inputs = [v0, v1, v2]
    outputs = ConcatFromSequence(opset_version, axis=axis).run(inputs)
    check("ConcatFromSequence", {"axis": axis}, [inputs], outputs, opset_version)


def test_concatfromsequence_01():
    opset_version = 13
    axis = 1
    new_axis = 1
    v0 = np.random.randn(2, 3).astype(np.float32)
    v1 = np.random.randn(2, 3).astype(np.float32)
    v2 = np.random.randn(2, 3).astype(np.float32)

    inputs = [v0, v1, v2]
    outputs = ConcatFromSequence(opset_version, axis=axis, new_axis=new_axis).run(inputs)
    check("ConcatFromSequence", {"axis": axis, "new_axis": new_axis}, [inputs], outputs, opset_version)
