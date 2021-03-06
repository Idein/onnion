import numpy as np
from onnion_runtime import ArgMax

from .utils import check


def test_argmax_00():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_01():
    opset_version = 13
    keepdims = 1
    attrs = {"keepdims": keepdims}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_02():
    opset_version = 13
    keepdims = 1
    select_last_index = 1
    attrs = {"keepdims": keepdims, "select_last_index": select_last_index}

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_03():
    opset_version = 13
    keepdims = 1
    select_last_index = 1
    attrs = {"keepdims": keepdims, "select_last_index": select_last_index}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_04():
    opset_version = 13
    axis = 1
    keepdims = 1
    attrs = {"axis": axis, "keepdims": keepdims}

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_05():
    opset_version = 13
    axis = 1
    keepdims = 1
    attrs = {"axis": axis, "keepdims": keepdims}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_06():
    opset_version = 13
    axis = 1
    keepdims = 1
    select_last_index = 1
    attrs = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_07():
    opset_version = 13
    axis = 1
    keepdims = 1
    select_last_index = 1
    attrs = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_08():
    opset_version = 13
    axis = -1
    keepdims = 1
    attrs = {"axis": axis, "keepdims": keepdims}

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_09():
    opset_version = 13
    axis = -1
    keepdims = 1
    attrs = {"axis": axis, "keepdims": keepdims}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_10():
    opset_version = 13
    axis = -1
    keepdims = 1
    select_last_index = 1
    attrs = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_11():
    opset_version = 13
    axis = -1
    keepdims = 1
    select_last_index = 1
    attrs = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_12():
    opset_version = 13
    axis = 1
    keepdims = 0
    attrs = {"axis": axis, "keepdims": keepdims}

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_13():
    opset_version = 13
    axis = 1
    keepdims = 0
    attrs = {"axis": axis, "keepdims": keepdims}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_14():
    opset_version = 13
    axis = 1
    keepdims = 0
    select_last_index = 1
    attrs = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)


def test_argmax_15():
    opset_version = 13
    axis = 1
    keepdims = 0
    select_last_index = 1
    attrs = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    inputs = [data]

    check(ArgMax, opset_version, attrs, inputs)
