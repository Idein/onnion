import numpy as np
from onnion_runtime import Mod

from .utils import check


def test_mod_00():
    opset_version = 13
    attrs = dict()

    x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
    y = np.array([7]).astype(np.int32)
    inputs = [x, y]

    check(Mod, opset_version, attrs, inputs)


def test_mod_01():
    opset_version = 13
    attrs = {"fmod": 1}

    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
    inputs = [x, y]

    check(Mod, opset_version, attrs, inputs)


def test_mod_02():
    opset_version = 13
    attrs = {"fmod": 1}

    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
    inputs = [x, y]

    check(Mod, opset_version, attrs, inputs)
