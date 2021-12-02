import numpy as np
from onnion_runtime import DepthToSpace

from .utils import check


def test_depthtospace_00():
    opset_version = 13
    blocksize = 2
    attrs = {"blocksize": blocksize}

    x = np.array(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
            ]
        ]
    ).astype(np.float32)
    inputs = [x]

    check(DepthToSpace, opset_version, attrs, inputs)


def test_depthtospace_01():
    opset_version = 13
    blocksize = 2
    mode = "CRD"
    attrs = {"blocksize": blocksize, "mode": mode}

    x = np.array(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
            ]
        ]
    ).astype(np.float32)
    inputs = [x]

    check(DepthToSpace, opset_version, attrs, inputs)


def test_depthtospace_02():
    opset_version = 10
    blocksize = 2
    attrs = {"blocksize": blocksize}

    x = np.array(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
            ]
        ]
    ).astype(np.float32)
    inputs = [x]

    check(DepthToSpace, opset_version, attrs, inputs)
