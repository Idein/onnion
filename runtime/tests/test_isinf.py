import numpy as np
from onnion_runtime import IsInf

from .utils import check


def test_isinf_00():
    opset_version = 13
    attrs = dict()

    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
    inputs = [x]

    check(IsInf, opset_version, attrs, inputs)


def test_isinf_01():
    opset_version = 13
    detect_positive = 0
    attrs = {"detect_positive": detect_positive}

    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
    inputs = [x]

    check(IsInf, opset_version, attrs, inputs)


def test_isinf_02():
    opset_version = 13
    detect_negative = 0
    attrs = {"detect_negative": detect_negative}

    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
    inputs = [x]

    check(IsInf, opset_version, attrs, inputs)
