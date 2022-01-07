import numpy as np


class LogSoftmax:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        if self.version >= 13:
            self.axis = kwargs.get("axis", -1)
        else:
            self.axis = kwargs.get("axis", 1)

    def run(self, x):
        if self.version >= 13:
            return [logsoftmax(x, axis=self.axis)]
        else:
            shape0 = x.shape
            n = int(np.product(shape0[: self.axis]))
            d = int(np.product(shape0[self.axis :]))
            shape1 = (n, d)
            result = logsoftmax_2d(x.reshape(shape1))
            return [result.reshape(shape0)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/65974860e20b311d14b642ce22b5a56b8c176ca5/onnx/backend/test/case/node/logsoftmax.py#L15-L19
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def logsoftmax(x, axis=-1):  # type: (np.ndarray, int) -> np.ndarray
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return (x - x_max) - np.log(s)


# The following code has been copied from
# https://github.com/onnx/onnx/blob/7988d8360b11e6003560076e9b1d4aa426db3244/onnx/backend/test/case/node/logsoftmax.py#L30-L33
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def logsoftmax_2d(x):  # type: (np.ndarray) -> np.ndarray
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))
