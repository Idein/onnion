import numpy as np
from onnion_runtime import Celu

from .utils import check


def test_celu_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Celu(opset_version).run(x)

    check("Celu", dict(), [x], outputs, opset_version)


def test_celu_01():
    opset_version = 13

    alpha = 2.0
    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Celu(opset_version, alpha=alpha).run(x)

    check("Celu", {"alpha": alpha}, [x], outputs, opset_version)
