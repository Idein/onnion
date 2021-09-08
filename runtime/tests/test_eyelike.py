import numpy as np
import onnx
from onnion_runtime import EyeLike

from .utils import check


def test_eyelike_00():
    opset_version = 13

    k = 1
    dtype = int(onnx.TensorProto.FLOAT)
    x = np.random.randint(0, 100, size=(4, 5), dtype=np.int32)
    outputs = EyeLike(opset_version, k=k, dtype=dtype).run(x)

    check("EyeLike", {"k": k, "dtype": dtype}, [x], outputs, opset_version)


def test_eyelike_01():
    opset_version = 13

    dtype = int(onnx.TensorProto.DOUBLE)
    x = np.random.randint(0, 100, size=(3, 4), dtype=np.int32)
    outputs = EyeLike(opset_version, dtype=dtype).run(x)

    check("EyeLike", {"dtype": dtype}, [x], outputs, opset_version)


def test_eyelike_02():
    opset_version = 13

    x = np.random.randint(0, 100, size=(3, 4), dtype=np.int32)
    outputs = EyeLike(opset_version).run(x)

    check("EyeLike", dict(), [x], outputs, opset_version)
