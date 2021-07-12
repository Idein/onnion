import numpy as np
from onnion_runtime import Exp

from .utils import check


def exp_checker(opset_version):
    v0 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    [v1] = Exp(opset_version).run(v0)

    check("Exp", dict(), {"v0": v0}, {"v1": v1}, opset_version)


def test_exp_00():
    exp_checker(9)
