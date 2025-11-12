import numpy as np

# import pytest
from onnion_runtime import Resize

from .utils import check


def test_resize_00():
    opset_version = 13
    attrs = {"mode": "cubic"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_01():
    opset_version = 13
    attrs = {"mode": "cubic", "cubic_coeff_a": -0.5, "exclude_outside": True}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


# @pytest.mark.xfail
def test_resize_02():
    opset_version = 13
    attrs = {"mode": "cubic", "coordinate_transformation_mode": "align_corners"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_03():
    opset_version = 13
    attrs = {"mode": "linear"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


# @pytest.mark.xfail
def test_resize_04():
    opset_version = 13
    attrs = {"mode": "linear", "coordinate_transformation_mode": "align_corners"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_05():
    opset_version = 13
    attrs = {"mode": "nearest"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_06():
    opset_version = 13
    attrs = {"mode": "cubic"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_07():
    opset_version = 13
    attrs = {"mode": "linear", "coordinate_transformation_mode": "pytorch_half_pixel"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 3, 1], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_08():
    opset_version = 13
    attrs = {"mode": "nearest"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 1, 3], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_09():
    opset_version = 13
    attrs = {"mode": "linear", "coordinate_transformation_mode": "tf_crop_and_resize"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)
    inputs = [x, roi, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_10():
    opset_version = 13
    attrs = {"mode": "linear", "coordinate_transformation_mode": "tf_crop_and_resize", "extrapolation_value": 10.0}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)
    inputs = [x, roi, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_11():
    opset_version = 13
    attrs = {"mode": "cubic"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_12():
    opset_version = 13
    attrs = {"mode": "cubic", "cubic_coeff_a": -0.5, "exclude_outside": True}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_13():
    opset_version = 13
    attrs = {"mode": "cubic", "coordinate_transformation_mode": "align_corners"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_14():
    opset_version = 13
    attrs = {"mode": "cubic", "coordinate_transformation_mode": "asymmetric"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_15():
    opset_version = 13
    attrs = {"mode": "linear"}

    x = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_16():
    opset_version = 13
    attrs = {"mode": "linear", "coordinate_transformation_mode": "align_corners"}

    x = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_17():
    opset_version = 13
    attrs = {"mode": "nearest"}

    x = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    inputs = [x, None, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_18():
    opset_version = 13
    attrs = {"mode": "cubic"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 9, 10], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_19():
    opset_version = 13
    attrs = {"mode": "nearest"}

    x = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 7, 8], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_20():
    opset_version = 13
    attrs = {"mode": "nearest", "coordinate_transformation_mode": "half_pixel", "nearest_mode": "ceil"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 8, 8], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_21():
    opset_version = 13
    attrs = {"mode": "nearest", "coordinate_transformation_mode": "align_corners", "nearest_mode": "floor"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 8, 8], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_22():
    opset_version = 13
    attrs = {"mode": "nearest", "coordinate_transformation_mode": "asymmetric", "nearest_mode": "round_prefer_ceil"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )
    sizes = np.array([1, 1, 8, 8], dtype=np.int64)
    inputs = [x, None, None, sizes]

    check(Resize, opset_version, attrs, inputs)


def test_resize_23():
    opset_version = 10
    attrs = {"mode": "linear"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    inputs = [x, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_24():
    opset_version = 10
    attrs = {"mode": "nearest"}

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    inputs = [x, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_25():
    opset_version = 10
    attrs = {"mode": "linear"}

    x = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    inputs = [x, scales]

    check(Resize, opset_version, attrs, inputs)


def test_resize_26():
    opset_version = 10
    attrs = {"mode": "nearest"}

    x = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    inputs = [x, scales]

    check(Resize, opset_version, attrs, inputs)
