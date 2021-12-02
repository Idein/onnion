import numpy as np
from onnion_runtime import DynamicQuantizeLinear

from .utils import check


def test_dynamicquantizelinear_00():
    opset_version = 13
    attrs = dict()

    x = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
    inputs = [x]

    check(DynamicQuantizeLinear, opset_version, attrs, inputs)


def test_dynamicquantizelinear_01():
    opset_version = 13
    attrs = dict()

    x = np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0]).astype(np.float32)
    inputs = [x]

    check(DynamicQuantizeLinear, opset_version, attrs, inputs)


def test_dynamicquantizelinear_02():
    opset_version = 13
    attrs = dict()

    x = np.array([1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345]).astype(np.float32).reshape((3, 4))
    inputs = [x]

    check(DynamicQuantizeLinear, opset_version, attrs, inputs)
