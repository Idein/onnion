import numpy as np
from onnion_runtime import Elu

from .utils import check


def test_elu_00():
    opset_version = 13

    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Elu(opset_version).run(x)

    check("Elu", dict(), [x], outputs, opset_version)


def test_elu_01():
    opset_version = 13

    alpha = 2.0
    x = np.random.randn(3, 4, 5).astype(np.float32)
    outputs = Elu(opset_version, alpha=alpha).run(x)

    check("Elu", {"alpha": alpha}, [x], outputs, opset_version)
