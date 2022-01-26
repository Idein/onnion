import pytest
from onnion_runtime import RandomNormal

from .utils import check


@pytest.mark.xfail(raises=AssertionError)
def test_randomnormal_00():
    opset_version = 13
    attrs = {"dtype": 1, "mean": 0.0, "scale": 1.0, "seed": 0.0, "shape": [2, 3]}

    inputs = []

    check(RandomNormal, opset_version, attrs, inputs)


@pytest.mark.xfail(raises=AssertionError)
def test_randomnormal_01():
    opset_version = 13
    attrs = {"shape": [2, 3]}

    inputs = []

    check(RandomNormal, opset_version, attrs, inputs)
