import numpy as np
from onnion_runtime import EyeLike

from .utils import check


def test_eyelike_00():
    opset_version = 13
    k = 1
    dtype = 1  # FLOAT
    attrs = {"k": k, "dtype": dtype}

    x = np.random.randint(0, 100, size=(4, 5), dtype=np.int32)
    inputs = [x]

    check(EyeLike, opset_version, attrs, inputs)


def test_eyelike_01():
    opset_version = 13
    dtype = 11  # DOUBLE
    attrs = {"dtype": dtype}

    x = np.random.randint(0, 100, size=(3, 4), dtype=np.int32)
    inputs = [x]

    check(EyeLike, opset_version, attrs, inputs)


def test_eyelike_02():
    opset_version = 13
    attrs = dict()

    x = np.random.randint(0, 100, size=(3, 4), dtype=np.int32)
    inputs = [x]

    check(EyeLike, opset_version, attrs, inputs)
