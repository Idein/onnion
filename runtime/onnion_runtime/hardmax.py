import numpy as np


class Hardmax:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", -1)

    def run(self, x):
        return [hardmax(x, axis=self.axis)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/f3843b26e23c1e660e2990c3b74fda5bf6ba4c8c/onnx/backend/test/case/node/hardmax.py#L15-L19
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def hardmax(x, axis=-1):  # type: (np.ndarray, int) -> np.ndarray
    x_argmax = np.argmax(x, axis=axis)
    y = np.zeros_like(x)
    np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
    return y
