import numpy as np
from onnion_runtime import ConstantOfShape
from onnx import numpy_helper

from .utils import check


def test_constantofshape_00():
    opset_version = 13

    v = np.array([1]).astype(np.float32)
    shape = np.array([4, 3, 2]).astype(np.int64)
    outputs = ConstantOfShape(opset_version, value=v).run(shape)

    check("ConstantOfShape", {"value": numpy_helper.from_array(v)}, [shape], outputs, opset_version)


def test_constantofshape_01():
    opset_version = 13

    v = np.array([0]).astype(np.int32)
    shape = np.array(
        [
            0,
        ]
    ).astype(np.int64)
    outputs = ConstantOfShape(opset_version, value=v).run(shape)

    check("ConstantOfShape", {"value": numpy_helper.from_array(v)}, [shape], outputs, opset_version)


def test_constantofshape_02():
    opset_version = 13

    v = np.array([0]).astype(np.int32)
    shape = np.array([10, 6]).astype(np.int64)
    outputs = ConstantOfShape(opset_version, value=v).run(shape)

    check("ConstantOfShape", {"value": numpy_helper.from_array(v)}, [shape], outputs, opset_version)
