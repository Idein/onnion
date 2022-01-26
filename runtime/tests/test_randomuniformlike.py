import numpy as np
import pytest
from onnion_runtime import RandomUniformLike

from .utils import check, on_arm32


@pytest.mark.skipif(on_arm32(), reason="need to pass tests on x86_64")
@pytest.mark.xfail(raises=AssertionError)
def test_randomuniformlike_00():
    opset_version = 13
    attrs = {"dtype": 1, "high": 5.0, "low": 3.0, "seed": 0.0}

    x = np.random.randn(2, 3).astype(np.float32)
    inputs = [x]

    check(RandomUniformLike, opset_version, attrs, inputs)


@pytest.mark.skipif(on_arm32(), reason="need to pass tests on x86_64")
@pytest.mark.xfail(raises=AssertionError)
def test_randomuniformlike_01():
    opset_version = 13
    attrs = dict()

    x = np.random.randn(2, 3).astype(np.float32)
    inputs = [x]

    check(RandomUniformLike, opset_version, attrs, inputs)
