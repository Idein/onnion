import numpy as np


class ArgMin:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 0)
        self.keepdims = kwargs.get("keepdims", 1)
        self.select_last_index = kwargs.get("select_last_index", 0)

    def run(self, data):
        if self.version >= 12:
            if self.select_last_index:
                return [argmin_use_numpy_select_last_index(data, axis=self.axis, keepdims=self.keepdims)]
            else:
                return [argmin_use_numpy(data, axis=self.axis, keepdims=self.keepdims)]
        else:
            return [argmin_use_numpy(data, axis=self.axis, keepdims=self.keepdims)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/c4de8b7487603c4496cf6d18b3836be111d0eb90/onnx/backend/test/case/node/argmin.py#L15
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def argmin_use_numpy(data, axis=0, keepdims=1):  # type: (np.ndarray, int, int) -> (np.ndarray)
    result = np.argmin(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


# The following code has been copied from
# https://github.com/onnx/onnx/blob/c4de8b7487603c4496cf6d18b3836be111d0eb90/onnx/backend/test/case/node/argmin.py#L22
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def argmin_use_numpy_select_last_index(data, axis=0, keepdims=True):  # type: (np.ndarray, int, int) -> (np.ndarray)
    data = np.flip(data, axis)
    result = np.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)
