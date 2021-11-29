import numpy as np
import pytest
from onnion_runtime import Cast

from .utils import check, on_arm32


def test_cast_00():
    opset_version = 13
    t = 1  # FLOAT
    attrs = {"to": t}

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(Cast, opset_version, attrs, inputs)


def test_cast_01():
    opset_version = 13
    t = 11  # DOUBLE
    attrs = {"to": t}

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(Cast, opset_version, attrs, inputs)


def test_cast_02():
    opset_version = 13
    t = 6  # INT32
    attrs = {"to": t}

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(Cast, opset_version, attrs, inputs)


@pytest.mark.xfail(on_arm32(), reason="astype returns different results on arm32")
def test_cast_03():
    opset_version = 13
    t = 12  # UINT32
    attrs = {"to": t}

    x = np.random.randn(3, 5).astype(np.float32)
    inputs = [x]

    check(Cast, opset_version, attrs, inputs)
