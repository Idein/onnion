import numpy as np
from onnion_runtime import LeakyRelu

from .utils import check


def test_leakyrelu_00():
    opset_version = 13
    x = np.random.randn(3, 5).astype(np.float32)
    outputs = LeakyRelu(opset_version).run(x)

    check("LeakyRelu", dict(), [x], outputs, opset_version)


def test_leakyrelu_01():
    opset_version = 13
    alpha = 2.0
    x = np.random.randn(3, 5).astype(np.float32)
    outputs = LeakyRelu(opset_version, alpha=alpha).run(x)

    check("LeakyRelu", {"alpha": alpha}, [x], outputs, opset_version)
