import numpy as np
from onnion_runtime import Cast
from onnx import TensorProto

from .utils import check


def test_cast_00():
    opset_version = 13
    t = int(TensorProto.FLOAT)
    x = np.random.randn(3, 5).astype(np.float32)
    y = Cast(opset_version, to=t).run(x)
    check("Cast", {"to": t}, [x], y, opset_version)


def test_cast_01():
    opset_version = 13
    t = int(TensorProto.DOUBLE)
    x = np.random.randn(3, 5).astype(np.float32)
    y = Cast(opset_version, to=t).run(x)
    check("Cast", {"to": t}, [x], y, opset_version)


def test_cast_02():
    opset_version = 13
    t = int(TensorProto.INT32)
    x = np.random.randn(3, 5).astype(np.float32)
    y = Cast(opset_version, to=t).run(x)
    check("Cast", {"to": t}, [x], y, opset_version)


def test_cast_03():
    opset_version = 13
    t = int(TensorProto.UINT32)
    x = np.random.randn(3, 5).astype(np.float32)
    y = Cast(opset_version, to=t).run(x)
    check("Cast", {"to": t}, [x], y, opset_version)
