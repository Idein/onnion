import numpy as np
from onnion_runtime import Tile

from .utils import check


def test_tile_00():
    opset_version = 13

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    inputs = [x, repeats]
    outputs = Tile(opset_version).run(x, repeats)

    check("Tile", dict(), inputs, outputs, opset_version)
