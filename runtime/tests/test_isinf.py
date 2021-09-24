import numpy as np
from onnion_runtime import IsInf

from .utils import check


def test_isinf_00():
    opset_version = 13

    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
    outputs = IsInf(opset_version).run(x)

    check("IsInf", dict(), [x], outputs, opset_version)


def test_isinf_01():
    opset_version = 13

    detect_positive = 0
    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
    outputs = IsInf(opset_version, detect_positive=detect_positive).run(x)

    check("IsInf", {"detect_positive": detect_positive}, [x], outputs, opset_version)


def test_isinf_02():
    opset_version = 13

    detect_negative = 0
    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
    outputs = IsInf(opset_version, detect_negative=detect_negative).run(x)

    check("IsInf", {"detect_negative": detect_negative}, [x], outputs, opset_version)
