import numpy as np
from onnion_runtime import Clip

from .utils import check


def test_clip_00():
    opset_version = 13
    attrs = dict()

    x = np.array([-2, 0, 2]).astype(np.float32)
    min_val = np.array(-1).astype(np.float32)
    max_val = np.array(1).astype(np.float32)
    inputs = [x, min_val, max_val]

    check(Clip, opset_version, attrs, inputs)


def test_clip_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    min_val = np.array(-1).astype(np.float32)
    max_val = np.array(1).astype(np.float32)
    inputs = [x, min_val, max_val]

    check(Clip, opset_version, attrs, inputs)


def test_clip_02():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    inputs = [x]

    check(Clip, opset_version, attrs, inputs)
