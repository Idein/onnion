import numpy as np


class Softmax:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        if self.version >= 13:
            self.axis = kwargs.get("axis", -1)
        else:
            self.axis = kwargs.get("axis", 1)

    def run(self, x):
        if self.version >= 13:
            return [softmax(x, axis=self.axis)]
        else:
            shape0 = x.shape
            n = int(np.product(shape0[: self.axis]))
            d = int(np.product(shape0[self.axis :]))
            shape1 = (n, d)
            result = softmax_2d(x.reshape(shape1))
            return [result.reshape(shape0)]


def softmax(x, axis):
    # The following code has been copied from
    # https://github.com/onnx/onnx/blob/547032edb8f86adb1b38d37f0a87aa61ee2ff580/onnx/backend/test/case/node/softmax.py#L13C1-L17C1
    # Copyrights (c) ONNX Project Contributers
    # License: Apache-2.0
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s


def softmax_2d(x):
    # The following code has been copied from
    # https://github.com/onnx/onnx/blame/3e59c7104822ba2aa2d414eb5dd2bdcad7ccb309/onnx/backend/test/case/node/softmax.py#L31-L33C66
    # Copyrights (c) ONNX Project Contributers
    # License: Apache-2.0
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
