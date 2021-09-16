import numpy as np
from onnion_runtime import HardSigmoid

from .utils import check


def test_hardsigmoid_00():
    opset_version = 13

    alpha = 0.5
    beta = 0.6
    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = HardSigmoid(opset_version, alpha=alpha, beta=beta).run(x)

    check("HardSigmoid", {"alpha": alpha, "beta": beta}, [x], outputs, opset_version)


def test_hardsigmoid_01():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = HardSigmoid(opset_version).run(x)

    check("HardSigmoid", dict(), [x], outputs, opset_version)
