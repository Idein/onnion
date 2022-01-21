import numpy as np


class OneHot:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", -1)

    def run(self, indices, depth, values):
        off_value = values[0]
        on_value = values[1]
        y = one_hot(indices, depth, axis=self.axis, dtype=values.dtype)
        return [y * (on_value - off_value) + off_value]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/4ae30cec896b08ad73ef0aab3539f36930a8c1e3/onnx/backend/test/case/node/onehot.py#L15-L26
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def one_hot(indices, depth, axis=-1, dtype=np.float32):  # type: ignore
    """Compute one hot from indices at a specific axis"""
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
    values = np.reshape(np.mod(values, depth), ls + (1,) + rs)
    return np.asarray(targets == values, dtype=dtype)
