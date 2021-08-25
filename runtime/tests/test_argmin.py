import numpy as np
from onnion_runtime import ArgMin

from .utils import check


def test_argmin_00():
    opset_version = 13

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    keepdims = 1
    outputs = ArgMin(opset_version, keepdims=keepdims).run(data)

    check("ArgMin", {"keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_01():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    keepdims = 1
    outputs = ArgMin(opset_version, keepdims=keepdims).run(data)

    check("ArgMin", {"keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_02():
    opset_version = 13

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    keepdims = 1
    select_last_index = 1
    outputs = ArgMin(opset_version, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check("ArgMin", {"keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version)


def test_argmin_03():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    keepdims = 1
    select_last_index = 1
    outputs = ArgMin(opset_version, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check("ArgMin", {"keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version)


def test_argmin_04():
    opset_version = 13

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    axis = 1
    keepdims = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims).run(data)

    check("ArgMin", {"axis": axis, "keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_05():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    axis = 1
    keepdims = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims).run(data)

    check("ArgMin", {"axis": axis, "keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_06():
    opset_version = 13

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    axis = 1
    keepdims = 1
    select_last_index = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check(
        "ArgMin", {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version
    )


def test_argmin_07():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    axis = 1
    keepdims = 1
    select_last_index = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check(
        "ArgMin", {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version
    )


def test_argmin_08():
    opset_version = 13

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    axis = -1
    keepdims = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims).run(data)

    check("ArgMin", {"axis": axis, "keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_09():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    axis = -1
    keepdims = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims).run(data)

    check("ArgMin", {"axis": axis, "keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_10():
    opset_version = 13

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    axis = -1
    keepdims = 1
    select_last_index = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check(
        "ArgMin", {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version
    )


def test_argmin_11():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    axis = -1
    keepdims = 1
    select_last_index = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check(
        "ArgMin", {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version
    )


def test_argmin_12():
    opset_version = 13

    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    axis = 1
    keepdims = 0
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims).run(data)

    check("ArgMin", {"axis": axis, "keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_13():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    axis = 1
    keepdims = 0
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims).run(data)

    check("ArgMin", {"axis": axis, "keepdims": keepdims}, [data], outputs, opset_version)


def test_argmin_14():
    opset_version = 13

    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    axis = 1
    keepdims = 0
    select_last_index = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check(
        "ArgMin", {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version
    )


def test_argmin_15():
    opset_version = 13

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    axis = 1
    keepdims = 0
    select_last_index = 1
    outputs = ArgMin(opset_version, axis=axis, keepdims=keepdims, select_last_index=select_last_index).run(data)

    check(
        "ArgMin", {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}, [data], outputs, opset_version
    )
