import numpy as np
from onnion_runtime import ScatterND

from .utils import check


def test_scatternd_00():
    opset_version = 13

    data = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array(
        [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]],
        dtype=np.float32,
    )
    inputs = [data, indices, updates]
    outputs = ScatterND(opset_version).run(data, indices, updates)

    check("ScatterND", dict(), inputs, outputs, opset_version)


def test_scatternd_01():
    opset_version = 13

    with open("legacy_tests/scatter_nd.npy", "rb") as f:
        data = np.load(f)
        indices = np.load(f)
        updates = np.load(f)
    print(indices)
    inputs = [data, indices, updates]
    outputs = ScatterND(opset_version).run(data, indices, updates)

    check("ScatterND", dict(), inputs, outputs, opset_version)
