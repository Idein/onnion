import numpy as np
from onnion_runtime import DequantizeLinear

from .utils import check


def test_dequantizelinear_00():
    opset_version = 13

    x = np.array(
        [
            [
                [[3, 89], [34, 200], [74, 59]],
                [[5, 24], [24, 87], [32, 13]],
                [[245, 99], [4, 142], [121, 102]],
            ],
        ],
        dtype=np.uint8,
    )
    x_scale = np.array([2, 4, 5], dtype=np.float32)
    x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    inputs = [x, x_scale, x_zero_point]
    outputs = DequantizeLinear(opset_version).run(x, x_scale, x_zero_point)

    check("DequantizeLinear", dict(), inputs, outputs, opset_version)


def test_dequantizelinear_01():
    opset_version = 13

    x = np.array([0, 3, 128, 255]).astype(np.uint8)
    x_scale = np.array(2).astype(np.float32)
    x_zero_point = np.array(128).astype(np.uint8)
    inputs = [x, x_scale, x_zero_point]
    outputs = DequantizeLinear(opset_version).run(x, x_scale, x_zero_point)

    check("DequantizeLinear", dict(), inputs, outputs, opset_version)
