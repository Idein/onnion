import pytest
from onnion_runtime import RandomNormal

from .utils import check, on_arm32


@pytest.mark.skipif(on_arm32(), reason="need to pass tests on x86_64")
@pytest.mark.xfail(raises=AssertionError)
def test_randomnormal_00():
    opset_version = 13
    attrs = {"dtype": 1, "mean": 5.0, "scale": 3.0, "seed": 0.0, "shape": [2, 3]}

    inputs = []

    check(RandomNormal, opset_version, attrs, inputs)


@pytest.mark.skipif(on_arm32(), reason="need to pass tests on x86_64")
@pytest.mark.xfail(raises=AssertionError)
def test_randomnormal_01():
    opset_version = 13
    attrs = {"shape": [2, 3]}

    inputs = []

    check(RandomNormal, opset_version, attrs, inputs)
