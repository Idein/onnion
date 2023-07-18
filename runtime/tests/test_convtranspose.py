import numpy as np
from onnion_runtime import ConvTranspose

from .utils import check


def test_convtranspose_00() -> None:
    opset_version = 13
    attrs = dict()
    x = np.array([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]).astype(np.float32)  # (1, 1, 3, 3)

    W = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    ).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


def test_convtranspose_01() -> None:
    opset_version = 13
    attrs = {"strides": [3, 2], "output_padding": [1, 1]}

    x = np.array([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]).astype(np.float32)  # (1, 1, 3, 3)

    W = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    ).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


# test dillation
def test_convtranspose_02() -> None:
    opset_version = 13
    attrs = {"dilations": [2, 2]}
    x = np.random.randn(1, 1, 3, 3).astype(np.float32)
    W = np.random.randn(1, 1, 2, 2).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


# test pads
def test_convtranspose_03() -> None:
    opset_version = 13
    attrs = {"strides": [3, 2], "pads": [1, 2, 1, 2]}
    x = np.random.randn(1, 1, 3, 3).astype(np.float32)
    W = np.random.randn(1, 2, 3, 3).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


# specify output shape
def test_convtranspose_04() -> None:
    opset_version = 13
    attrs = {"strides": [3, 2], "output_shape": [10, 8]}
    x = np.random.randn(1, 1, 3, 3).astype(np.float32)
    W = np.random.randn(1, 2, 3, 3).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


# specify output shape and output padding
def test_convtranspose_05() -> None:
    opset_version = 13
    attrs = {"strides": [3, 2], "output_shape": [10, 8], "kernel_shape": [3, 3], "output_padding": [1, 1]}
    x = np.random.randn(1, 1, 3, 3).astype(np.float32)
    W = np.random.randn(1, 2, 3, 3).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


# larger channel number
def test_convtranspose_06() -> None:
    opset_version = 13
    attrs = {"strides": [2, 2], "kernel_shape": [2, 2], "pads": [0, 0, 0, 0]}
    x = np.random.randn(2, 24, 12, 12).astype(np.float32)
    W = np.random.randn(24, 24, 2, 2).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])


# opset 1
def test_convtranspose_07() -> None:
    opset_version = 1
    attrs = {"strides": [3, 2], "output_shape": [10, 8], "kernel_shape": [3, 3], "output_padding": [1, 1]}
    x = np.random.randn(1, 1, 3, 3).astype(np.float32)
    W = np.random.randn(1, 2, 3, 3).astype(np.float32)

    check(ConvTranspose, opset_version, attrs, [x, W])
