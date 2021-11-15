import numpy as np
from onnion_runtime import Abs

from .utils import check_by_data, check_by_onnxruntime, load_test_data, on_arm32, save_test_data


def test_abs_00():
    opset_version = 13

    if on_arm32():
        inputs, outputs0 = load_test_data()
    else:
        x = np.random.randn(3, 4, 5).astype(np.float32)
        inputs = [x]

    outputs = Abs(opset_version).run(*inputs)

    if on_arm32():
        check_by_data(outputs0, outputs)
    else:
        outputs0 = check_by_onnxruntime("Abs", dict(), inputs, outputs, opset_version)
        save_test_data(inputs, outputs0)
