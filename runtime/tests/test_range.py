import numpy as np
from onnion_runtime import Range

from .utils import check


def test_range_00():
    opset_version = 13
    attrs = dict()

    start = np.array(1).astype(np.float32)
    limit = np.array(5).astype(np.float32)
    delta = np.array(2).astype(np.float32)
    inputs = [start, limit, delta]

    check(Range, opset_version, attrs, inputs)


def test_range_01():
    opset_version = 13
    attrs = dict()

    start = np.array(10).astype(np.int32)
    limit = np.array(6).astype(np.int32)
    delta = np.array(-3).astype(np.int32)
    inputs = [start, limit, delta]

    check(Range, opset_version, attrs, inputs)
