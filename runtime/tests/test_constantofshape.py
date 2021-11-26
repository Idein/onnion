import numpy as np
from onnion_runtime import ConstantOfShape

from .utils import check


def test_constantofshape_00():
    opset_version = 13
    v = np.array([1]).astype(np.float32)
    attrs = {"value": v}

    shape = np.array([4, 3, 2]).astype(np.int64)
    inputs = [shape]

    check(ConstantOfShape, opset_version, attrs, inputs)


def test_constantofshape_01():
    opset_version = 13
    v = np.array([0]).astype(np.int32)
    attrs = {"value": v}

    shape = np.array([0]).astype(np.int64)
    inputs = [shape]

    check(ConstantOfShape, opset_version, attrs, inputs)


def test_constantofshape_02():
    opset_version = 13
    v = np.array([0]).astype(np.int32)
    attrs = {"value": v}

    shape = np.array([10, 6]).astype(np.int64)
    inputs = [shape]

    check(ConstantOfShape, opset_version, attrs, inputs)
