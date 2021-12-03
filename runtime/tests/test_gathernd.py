import numpy as np
from onnion_runtime import GatherND

from .utils import check


def test_gathernd_00():
    opset_version = 13
    attrs = dict()

    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
    inputs = [data, indices]

    check(GatherND, opset_version, attrs, inputs)


def test_gathernd_01():
    opset_version = 13
    attrs = dict()

    data = np.array([[0, 1], [2, 3]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    inputs = [data, indices]

    check(GatherND, opset_version, attrs, inputs)


def test_gathernd_02():
    opset_version = 13
    batch_dims = 1
    attrs = {"batch_dims": batch_dims}

    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    indices = np.array([[1], [0]], dtype=np.int64)
    inputs = [data, indices]

    check(GatherND, opset_version, attrs, inputs)
